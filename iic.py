import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class IDDLoss(nn.Module):
    def __init__(self, lamb=1.0, eps=sys.float_info.epsilon):
        super().__init__()
        self.lamb = lamb
        self.epsilon = eps

    def forward(self, x_out, x_tf_out):
        _, k = x_out.size()
        p_i_j = self._compute_joint(x_out, x_tf_out)
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        # but should be same, symmetric
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

        # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
        p_i_j[p_i_j < self.epsilon] = self.epsilon
        p_j[p_j < self.epsilon] = self.epsilon
        p_i[p_i < self.epsilon] = self.epsilon

        loss = torch.log(p_i_j)
        loss = loss - self.lamb * torch.log(p_j)
        loss = loss - self.lamb * torch.log(p_i)
        loss = - p_i_j * loss

        loss = loss.sum()

        return loss

    def _compute_joint(self, x_out, x_tf_out):
        bn, k = x_out.size()
        assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

        p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
        p_i_j = p_i_j.sum(dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j


class IICDataLoader:
    def __init__(self, reference_dl, tf_dl):
        assert isinstance(reference_dl, DataLoader)
        assert isinstance(tf_dl, list) and tf_dl != []

        reference_batch_size = reference_dl.batch_size
        if not all((reference_batch_size == x.batch_size for x in tf_dl)):
            raise Exception("All batch sizes must be equal")

        self.reference = reference_dl
        self.tf_list = tf_dl

    def __iter__(self):
        return IICDataLoader.iic_iterator(self.reference, self.tf_list)

    class iic_iterator:
        def __init__(self, reference_dl, tf_dl):
            self.reference_dl = reference_dl
            self.tf_dl = tf_dl
            self.iic_batch_size = reference_dl.batch_size * len(tf_dl)
            self.generator = zip(reference_dl, *tf_dl)

            sample = next(iter(reference_dl))[0]
            self.sample_shape = list(sample.shape)
            self.torch = sample.new_empty([])

            # The same tensor will be reused at each iteration
            self.original_batch = self.torch.new_zeros(self.iic_batch_size,
                                                       *self.sample_shape)
            self.tf_batch = self.torch.new_zeros(self.iic_batch_size,
                                                 *self.sample_shape)

        def __next__(self):
            tups = next(self.generator)
            original = tups[0][0]
            tf_tups = tups[1:]  # transformed data

            tf_data = [tup[0] for tup in tf_tups]
            tf_data = torch.cat(tf_data, dim=0, out=self.tf_batch)

            tf_batch_size = tf_data.size(0)
            original_batch_size = original.size(0)

            repetitions = tf_batch_size // original_batch_size
            original_repeated = torch.cat([original]*repetitions, dim=0,
                                          out=self.original_batch)

            orig_repet_batch_size = original_repeated.size(0)
            # The two batches must have the same size
            batch_size = min(tf_batch_size, orig_repet_batch_size)

            return original_repeated[:batch_size], self.tf_batch[:batch_size]


def matches(predictions, targets, net_k, gt_k):
    '''Find match between predictions cluster and target (ground truth)
    clusters.

    Parameters
    ----------
        predictions : Torch.Tensor
            Tensor (dim=1) with cluster prediction for each sample.
        targets : Torch.Tensor
            tensor (dim=1) with ground truth prediction for each sample.
        net_k : int
            number of net output clusters
        gt_k : int
            number of ground truth clusters

    Return
    ------
        list
            a list with length `net_k` that maps from `net` output cluster to
            ground truth clusters.
    '''
    assert predictions.shape == targets.shape
    assert predictions.dim() == 1 and targets.dim() == 1

    pred_to_target = [0 for _ in range(net_k)]
    pred_to_target_scores = [-1. for _ in range(net_k)]
    for i in range(net_k):
        for j in range(gt_k):
            score = (predictions == i) & (targets == j)
            score = score.sum()
            if score > pred_to_target_scores[i]:
                pred_to_target[i] = j
                pred_to_target_scores[i] = score.item()

    return pred_to_target


def predictions_list(net, dataloader, device):
    '''Returns a list of predictions tensors for each sub-head.

    Parameters
    ----------
        net : Pytorch network
        dataloader : Pytorch dataloader
        device : net's device

    Return
    ------
        list
            list containing all predictions for each sub-head
    '''
    predictions_list = []
    with torch.no_grad():
        batches_predictions = []
        for batch in dataloader:
            tf = batch[0]
            tf = tf.to(device)
            pred = net(tf)
            pred = [x.argmax(dim=1) for x in pred]
            batches_predictions.append(pred)

        for head_pred in zip(*batches_predictions):
            head_pred = torch.cat(head_pred)
            predictions_list.append(head_pred)

    return predictions_list


def train_loop(net, epochs, iic_dl, optimizer, loss_fn, device,
               print_log=False, callback=None):
    assert isinstance(loss_fn, IDDLoss)
    assert isinstance(iic_dl, IICDataLoader)

    print("Starting training")
    for i in range(epochs):
        net.train()
        if print_log:
            print(f"Epoch: {i}")

        start_time = time.time()
        for original, transformed in iic_dl:
            original = original.to(device)
            transformed = transformed.to(device)

            optimizer.zero_grad()

            out_original = net(original)
            out_tf = net(transformed)

            loss = [loss_fn(i, j) for i, j in zip(out_original, out_tf)]
            avg_loss = sum(loss) / len(loss)
            avg_loss.backward()

            optimizer.step()
        end_time = time.time()

        if print_log:
            print(f"Loss: {avg_loss.item()}")
            print(f"Elapsed time: {end_time - start_time} seconds")

        if callback is not None:
            with torch.no_grad():
                callback_return = callback()
            if callback_return is not None:
                break
