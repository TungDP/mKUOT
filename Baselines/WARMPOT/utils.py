import pdb
import torchvision

import torch
import torch.utils.data
import numpy as np
import torch.nn as nn

class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    Taken from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/sampler.py
    """

    def __init__(self, labels, batch_size):
        self.classes = sorted(set(labels.numpy()))

        n_classes = len(self.classes)
        self._n_samples = batch_size // n_classes
        self._n_remain = batch_size % n_classes
        if self._n_samples == 0:
            raise ValueError(f"batch_size should be bigger than the number of classes,\
                got {batch_size}")

        self._class_iters = [
            InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_) \
                for class_ in self.classes
        ]

        self.n_dataset = len(labels)
        self._n_batches = int(np.round(self.n_dataset // batch_size))
        if self._n_batches == 0:
            raise ValueError(f"Dataset is not big enough to generate batches with size\
                 {batch_size}")

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            add_class = set(np.random.choice(
                self.classes, self._n_remain, replace=False
            ))
            for class_iter in self._class_iters:
                if class_iter.class_ in add_class:
                    add_samples = 1
                else:
                    add_samples = 0
                indices.extend(class_iter.get(self._n_samples + add_samples))

            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_iters:
            class_iter.reset()

    def __len__(self):
        return self._n_batches


class InfiniteSliceIterator:
    def __init__(self, array, class_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.class_ = class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            print(f"there are really few items in class {self.class_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i : self.i]

class CrossEntropyWeighted(nn.Module):

    def __init__(self):
        super(CrossEntropyWeighted, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, weight):

        log_probs = self.logsoftmax(inputs)
        loss = (- targets * log_probs).sum(dim=1)

        weight_ = weight / (torch.sum(weight) + 1e-5)
        return torch.sum(weight_*loss)


def Entropy(logits):
	# logits BxN
	min_real = torch.finfo(logits.dtype).min
	logits = torch.clamp(logits, min=min_real)
	logits = logits - logits.logsumexp(dim=-1, keepdim=True)
	probs = torch.exp(logits)
	p_log_p = logits * probs

	return -p_log_p.sum(-1)