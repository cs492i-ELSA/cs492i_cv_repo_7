import re
import argparse
import os
import shutil
import time
import math
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
from ImageDataLoader import TwoStreamBatchSampler, PLTrainLoader
import copy

# Generates dataset for training using label propagation
def createLPTrainLoader(train_transformation,
                        weak_transformation,
                        datadir,
                        train_ids,
                        args,
                        uda=False,
                        uda_transformation=None):

    dataset = PLTrainLoader(datadir, 265,  transform=train_transformation, ids=train_ids, uda=uda, transform2=uda_transformation)
    vivid_dataset = PLTrainLoader(datadir, 265,  transform=weak_transformation, ids=train_ids)
    labeled_idxs, unlabeled_idxs = copy.deepcopy(dataset.labeled_idx), copy.deepcopy(dataset.unlabeled_idx)
    batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=0,
                                               pin_memory=True)
    train_loader_noshuff = torch.utils.data.DataLoader(vivid_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers= 0,
        pin_memory=True,
        drop_last=False)
    print("train loader size: {}".format(len(dataset.samples)))

    return train_loader, train_loader_noshuff, dataset

# Returns representation vectors for entire dataset.
def extractFeatures(loader,model):
    model.eval()
    embeddings = []

    for i, (batch_input, _, _, _) in enumerate(loader):
        X = batch_input
        X = torch.autograd.Variable(X.cuda())
        _, feats = model(X)
        embeddings.append(feats.data.cpu())

    embeddings = np.asarray(torch.cat(embeddings).numpy())
    return embeddings

# Set hyperparameters for finetuning
def loadFineTuningArgs(args):
    args.epoch_stg1 = 20
    args.epoch_stg2 = 10
    args.epoch_stg3 = 15
    args.batch_size = 36
    args.labeled_batch_size = 4
    args.basic_batch_size = 128
    args.lr = 0.002
    args.lr_stg1 = 0.05
    args.lr_stg2 = 0.02
    args.lr_stg3 = 0.002
    args.weight_decay = 2e-4
    args.dfs_k = 50

    return args