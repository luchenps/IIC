import pathlib
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import iic


class mnist_trunk(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = (5, 5)
        padding = (2, 2)

        pool_kernel_size = 2
        pool_stride = 2
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),

            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),

            nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),

            nn.Conv2d(256, 512, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, data):
        data = self.features(data)
        data = torch.flatten(data, start_dim=1)
        return data


class mnist_head(nn.Module):
    def __init__(self, input_dim, output_dim, sub_heads=5):
        super().__init__()
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Softmax(dim=1)
            ) for _ in range(sub_heads)
        ])

    def forward(self, data):
        out = [classifier(data) for classifier in self.classifiers]
        return out


class mnist_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = mnist_trunk()

        image_shape = 24
        x = torch.zeros(1, 1, image_shape, image_shape)
        with torch.no_grad():
            trunk_out = self.trunk(x)
            linear_input_dim = trunk_out.shape[-1]

        self.head = mnist_head(linear_input_dim, 25)

        self._initialize_weights()

    def forward(self, data):
        data = self.trunk(data)
        data = self.head(data)
        return data

    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode,
                                        nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def reference_transform():
    tf = (
        transforms.Compose([
            transforms.RandomChoice([
                transforms.CenterCrop((20, 20)),
                transforms.RandomCrop((20, 20))
            ]),
            transforms.Resize((24, 24)),
            transforms.ToTensor()
        ])
    )
    return tf


def random_transformation():
    tf = (
        transforms.Compose([
            transforms.RandomApply((
                transforms.RandomRotation((-25.0, 25.0)),
            )),
            transforms.RandomChoice([
                transforms.RandomCrop((16, 16)),
                transforms.RandomCrop((20, 20)),
                transforms.RandomCrop((24, 24)),
            ]),
            transforms.Resize((24, 24)),
            transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4],
                                   saturation=[0.6, 1.4], hue=[-0.125, 0.125]),
            transforms.ToTensor(),
        ])
    )
    return tf


def resize_transformation():
    tf = (
        transforms.Compose([
            transforms.CenterCrop((20, 20)),
            transforms.Resize((24, 24)),
            transforms.ToTensor()
        ])
    )
    return tf


def main(path):
    device = torch.device('cuda')
    print(f"Device: {device}")
    net = mnist_net().to(device)
    adam = optim.Adam(net.parameters(), lr=10**-4)
    MNIST = torchvision.datasets.MNIST

    train_ds = MNIST(path, train=True, transform=reference_transform())
    train_ds_tf = (MNIST(path, train=True, transform=random_transformation()) for i in range(5))
    eval_ds = MNIST(path, train=True, transform=resize_transformation())
    test_ds = MNIST(path, train=False, transform=resize_transformation())

    train_dl = DataLoader(train_ds, batch_size=70)
    train_dl_tf = [DataLoader(i, batch_size=70) for i in train_ds_tf]
    eval_dl = DataLoader(eval_ds, batch_size=256)
    test_dl = DataLoader(test_ds, batch_size=256)

    iic_dl = iic.IICDataLoader(train_dl, train_dl_tf)

    def batch_generator():
        return iic.iic_batch_generator(train_dl, train_dl_tf)

    train_targets = train_ds.targets.detach().clone().to(device)
    eval_targets = eval_ds.targets.detach().clone().to(device)
    best_acc = -1.
    selected_match = None
    best_head = -1

    def callback():
        '''Callback function to evaluate accuracy'''
        nonlocal best_acc
        nonlocal selected_match
        nonlocal best_head

        train_predictions = iic.predictions_list(net, train_dl, device)
        train_matches = []
        for prediction in train_predictions:
            match = iic.matches(prediction, train_targets, 25, 10)
            train_matches.append(match)

        eval_predictions = iic.predictions_list(net, eval_dl, device)
        eval_accs = []
        for match, prediction in zip(train_matches, eval_predictions):
            # make translations
            reordered = map(lambda i: match[i], prediction)
            reordered = list(reordered)
            reordered = torch.tensor(reordered).to(device)

            acc = (eval_targets == reordered).sum().to(torch.float32) / len(reordered)
            eval_accs.append(acc)

        eval_accs = torch.tensor(eval_accs)
        best_acc_arg = eval_accs.argmax()
        best_match = list(enumerate(train_matches[best_acc_arg]))
        print(f'''Accuracy
        MAX:  {eval_accs.max()}
        MIN:  {eval_accs.min()}
        MEAN: {eval_accs.mean()}
        STD:  {eval_accs.std()}
        BEST SUBHEAD: {eval_accs.argmax()}
        BEST MATCHES: {best_match}
        ''')

        if eval_accs.max() > best_acc:
            selected_match = train_matches[best_acc_arg]
            best_head = best_acc_arg

    iic.train_loop(net, 5, iic_dl, adam, iic.IDDLoss(), device,
                   print_log=True, callback=callback)

    test_predictions = iic.predictions_list(net, test_dl, device)
    best_prediction = test_predictions[best_head]
    best_prediction = best_prediction.detach().cpu()
    reordered_best_prediction = map(lambda i: selected_match[i],
                                    best_prediction)
    reordered_best_prediction = list(reordered_best_prediction)
    reordered_best_prediction = torch.tensor(reordered_best_prediction).to(device)
    test_targets = test_ds.targets.detach().clone().to(device)
    test_acc = (test_targets == reordered_best_prediction).sum().to(torch.float32) / len(test_targets)
    print(f"Test accuracy: {test_acc.item()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help="MNIST dataset path")
    args = parser.parse_args()
    path = pathlib.Path(args.path)
    assert path.exists()
    main(path)
