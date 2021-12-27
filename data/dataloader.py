import torch.utils.data
import numpy as np

from data.dataset import *
from data.sampler import *


def Dataset(args, mode):
    dataset = OmniglotDataset(mode=mode, root=args.dataset_path)
    n_classes = len(np.unique(dataset.y)) # labels
    if n_classes < args.classes_per_it_train or n_classes < args.classes_per_it_valid:
        raise (Exception('There are not enough classes in the dataset in order ' +
                         'to satisfy the chosen classes_per_it. Decrease the ' +
                         'classes_per_it_{tr/val} option and try again.'))
    return dataset


def Sampler(args, labels, mode):
    if 'train' in mode:
        classes_per_it = args.classes_per_it_train
        num_samples = args.num_support_train + args.num_query_train
    else:
        classes_per_it = args.classes_per_it_valid
        num_samples = args.num_support_valid + args.num_query_valid

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=args.iterations)


def Dataloader(args, mode):
    dataset = Dataset(args, mode)
    sampler = Sampler(args, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                             num_workers=args.workers, pin_memory=True)

    return dataloader

