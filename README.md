# IIC reimplementation for semisupervised clustering

This repository contains a reimplementation of [https://github.com/xu-ji/IIC]
(IIC code). For more information visit the original repository or read the
[https://arxiv.org/abs/1807.06653](IIC paper).

This reimplementation is an attempt to make a more portable code (between
different datasets and networks). It comes with a example code for MNIST (must
already be downloaded).

Even though this is not a fork it has some code from the original repository.

## Installing requirements
```
$ pip install -r requirements.txt
```

## Executing example
```
$ python mnist_example.py -p <path to mnist>
```

## Using IICDataloader
`IICDataloader` is an iterable that makes easier to use dataloaders with in the
training. It receives two parameters: a dataloader for `x` and a list of
dataloaders for each `gx` (see section 3.2 in the paper). All dataloaders must
have the same batch size e must not be shuffled.

## TODO
- More docstrings
- More example scripts
