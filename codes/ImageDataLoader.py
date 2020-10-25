from PIL import Image
import os
import os.path
import sys
import pdb
import faiss
from faiss import normalize_L2
import itertools
import scipy
import scipy.stats
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
import time
import pickle
import numpy as np

def defaultImageLoader(path):
    return Image.open(path).convert('RGB')

# Generates a pair of augmented image
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, inp):
        if self.transform is not None:
            out1 = self.transform(inp)
            out2 = self.transform(inp)
            return torch.stack([out1, out2])
        else:
            return inp, inp

# Iterator for unlabeled data
def instantIterator(iterable):
    return np.random.permutation(iterable)

# Iterator for labeled data
def eternalIterator(indices):
    def infiniteShuffle():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infiniteShuffle())

# Creates n-sized minibatch
def zipper(iterable, n):
    items = [iter(iterable)] * n
    return zip(*items)

# Generates a epoch where each unlabeled data appears once while labeled one appear as many as we need.
class TwoStreamSampler(Sampler):
    def __init__(self, primaryIndices, secondaryIndices, batchSize, secondaryBatchSize):
        self.primaryIndices = primaryIndices
        self.secondaryIndices = secondaryIndices
        self.secondaryBatchSize = secondaryBatchSize
        self.primaryBatchSize = batchSize - secondaryBatchSize
        print((self.primaryBatchSize), 'primary batch size')
        print(len(self.primaryIndices), 'primary indices')
        print((self.secondaryBatchSize), 'secondary batch size')
        print(len(self.secondaryIndices), 'secondary indices')
        
        assert len(self.primaryIndices) >= self.primaryBatchSize > 0
        assert len(self.secondaryIndices) >= self.secondaryBatchSize > 0

    def __iter__(self):
        primary_iter = instantIterator(self.primaryIndices)
        secondary_iter = eternalIterator(self.secondaryIndices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(zipper(primary_iter, self.primaryBatchSize),
                    zipper(secondary_iter, self.secondaryBatchSize))
        )

    def __len__(self):
        return len(self.primaryIndices) // self.primaryBatchSize

# Dataloader for managing label propagation
class PLTrainLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, num_class, ids=None, transform=None, loader=defaultImageLoader, uda=False, transform2=None):
        self.impath = os.path.join(rootdir, 'train/train_data')
        metaFile = os.path.join(rootdir, 'train/train_label')
        samples = []
        labeledIdx = []
        unlabeledIdx = []
        allLabels = []
        p_labels = []
        idx = 0
        if ids is not None:
            label_ids, unlabel_ids = ids
        with open(metaFile, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()    
                if ids is not None:
                    if (int(instance_id) in label_ids) and (int(label) == -1):
                        if (int(instance_id) not in unlabel_ids):
                            continue
                    if (int(instance_id) in unlabel_ids) and (int(label) != -1):
                        if (int(instance_id) not in label_ids):
                            continue
                if os.path.exists(os.path.join(self.impath, file_name)):
                    samples.append(file_name)
                    allLabels.append(int(label))
                    p_labels.append(-1)
                    if int(label) != -1:
                        labeledIdx.append(idx)
                    else:
                        unlabeledIdx.append(idx)
                    idx += 1

        self.transform = transform
        self.loader = loader
        self.samples = samples
        self.num_class = num_class

        self.imgs = self.samples
        self.pos_list = dict()
        self.pos_w = dict()
        self.labeledIdx = labeledIdx
        self.unlabeledIdx = unlabeledIdx
        self.allLabels = allLabels

        self.plabels = plabels
        self.p_weights = np.ones((len(self.imgs),))
        self.class_weights = np.ones((self.num_class,),dtype = np.float32)

        self.images_lists = [[] for i in range(self.num_class)]
        self.uda=uda
        self.transform2=transform2


    def __getitem__(self, index):
        filename = self.samples[index]
        sample = self.loader(os.path.join(self.impath, filename))

        if (index not in self.labeledIdx):
            target = self.plabels[index]
        else:
            target = self.allLabels[index]

        weight = self.p_weights[index]

        if self.transform is not None:
            if self.uda:
                sample = torch.stack([self.transform(sample), self.transform2(sample)])
            else:
                sample = self.transform(sample)
        
        c_weight = self.class_weights[target]

        return sample, target, weight, c_weight
        
    def __len__(self):
        return len(self.samples)

    # Conducts Label propagation using extracted representation vectors of entire data
    def updatePLabels(self, X, k = 50, max_iter = 20):
        print('Updating pseudo-labels...')
        alpha = 0.99
        labels = np.asarray(self.allLabels)
        labeledIdx = np.asarray(self.labeledIdx)
        unlabeledIdx = np.asarray(self.unlabeledIdx)

        
        # kNN search for the graph
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

        normalize_L2(X)
        index.add(X) 
        N = X.shape[0]
        Nidx = index.ntotal

        c = time.time()
        D, I = index.search(X, k + 1)
        elapsed = time.time() - c
        print('kNN Search done in %d seconds' % elapsed)

        # Create the graph
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis = 1)
        S[S==0] = 1
        D = np.array(1./ np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N, self.num_class))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
        for i in range(self.num_class):
            cur_idx = labeledIdx[np.where(labels[labeledIdx] ==i)]
            y = np.zeros((N,))
            y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:,i] = f

        # Handle numberical errors
        Z[Z < 0] = 0 

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z),1).numpy()
        probs_l1[probs_l1 <0] = 0
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(self.num_class)
        weights = weights / np.max(weights)
        plabels = np.argmax(probs_l1,1)

        plabels[labeledIdx] = labels[labeledIdx]
        weights[labeledIdx] = 1.0

        self.p_weights = weights.tolist()
        self.plabels = plabels

        # Compute the weight for each class
        for i in range(self.num_class):
            cur_idx = np.where(np.asarray(self.plabels) == i)[0]
            self.class_weights[i] = (float(labels.shape[0]) / self.num_class) / cur_idx.size

        return

# Basic data loader
class SimpleImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, unsym=False, transform=None, transform_base=None, loader=defaultImageLoader):
        if split == 'test':
            self.impath = os.path.join(rootdir, 'test_data')
            meta_file = os.path.join(self.impath, 'test_meta.txt')
        else:
            self.impath = os.path.join(rootdir, 'train/train_data')
            meta_file = os.path.join(rootdir, 'train/train_label')

        imnames = []
        imclasses = []
        
        with open(meta_file, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()        
                if int(label) == -1 and (split != 'unlabel' and split != 'test'):
                    continue
                if int(label) != -1 and (split == 'unlabel' or split == 'test'):
                    continue
                if (ids is None) or (int(instance_id) in ids):
                    if os.path.exists(os.path.join(self.impath, file_name)):
                        imnames.append(file_name)
                        if split == 'train' or split == 'val':
                            imclasses.append(int(label))

        self.transform = transform
        self.transform_base = transform_base
        self.TransformTwice = TransformTwice(transform)
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses
        self.unsym = unsym
    def __getitem__(self, index):
        filename = self.imnames[index]
        img = self.loader(os.path.join(self.impath, filename))
        
        if self.split == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.split != 'unlabel':
            if self.transform is not None:
                img = self.transform(img)
            label = self.imclasses[index]
            return img, label
        else:
            # For unlabeled data, returns a pair of augmented image for consistency regularization.        
            img1, img2 = img, img
            if self.unsym:
                if self.transform_base is not None:
                    img1 = self.transform_base(img)
                if self.transform is not None:
                    img2 = self.transform(img)
            else:
                img1, img2 = self.TransformTwice(img)
            return torch.stack([img1, img2])
        
    def __len__(self):
        return len(self.imnames)

