from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import numpy as np
import shutil
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision.models import *
from torchvision import datasets, models, transforms
import torch.nn.functional as F

from ImageDataLoader import SimpleImageLoader
from models import Res18, Res50, Dense121 , Res18_basic
from residual_attention_network import ResidualAttentionModel_92, ResidualAttentionModel_92_2
import glob
from model_resnet import ResidualNet
from attention_module import AttentionModule_stage1
import os
from bam import BAM
import argparse
#import torchlars

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

### LP-DeepSSL ###
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
from helpers import *
###
NUM_CLASSES = 265
NO_LABEL  = -1
if not IS_ON_NSML:
    DATASET_PATH = 'fashion_demo'

def top_n_accuracy_score(y_true, y_prob, n=5, normalize=True):
    num_obs, num_labels = y_prob.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_prob, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if normalize:
        return counter * 1.0 / num_obs
    else:
        return counter
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(opts, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opts.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
        
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, final_epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, opts.lambda_u * linear_rampup(epoch, final_epoch)


def split_ids(path, ratio):
    with open(path) as f:
        ids_l = []
        ids_u = []
        for i, line in enumerate(f.readlines()):
            if i == 0 or line == '' or line == '\n':
                continue
            line = line.replace('\n', '').split('\t')
            if int(line[1]) >= 0:
                ids_l.append(int(line[0]))
            else:
                ids_u.append(int(line[0]))

    ids_l = np.array(ids_l)
    ids_u = np.array(ids_u)

    perm = np.random.permutation(np.arange(len(ids_l)))
    cut = int(ratio*len(ids_l))
    train_ids = ids_l[perm][cut:]
    val_ids = ids_l[perm][:cut]

    return train_ids, val_ids, ids_u


### NSML functions
def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(root_path, 'test',
                               transform=transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.CenterCrop(opts.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])), batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True)
        print('loaded {} test images'.format(len(test_loader.dataset)))

    outputs = []
    s_t = time.time()
    for idx, image in enumerate(test_loader):
        if torch.cuda.is_available():
            image = image.cuda()
        probs, _ = model(image)
        output = torch.argmax(probs, dim=1)
        output = output.detach().cpu().numpy()
        outputs.append(output)

    outputs = np.concatenate(outputs)
    return outputs

def bind_nsml(model):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = model.module.state_dict()
        torch.save(state, os.path.join(dir_name, 'model.pt'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        model.load_state_dict(state)
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)


######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Sample Product200K Training')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
# basic settings
parser.add_argument('--name',default='ResA_Stg2', type=str, help='output model name')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--seed', type=int, default=123, help='random seed')

# basic hyper-parameters
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help=' ')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 5e-5)')
parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')
parser.add_argument('--lossXent', type=float, default=1, help='lossWeight for Xent')
parser.add_argument('--isMT', default=False, action='store_true', help='lossWeight for Xent')
parser.add_argument('--dataset', default='', type=str, help='lossWeight for Xent')

# arguments for logging and backup
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--save_epoch', type=int, default=50, help='saving epoch interval')

# arguments for UDA, LP
parser.add_argument('--isUDA', default=True, type=bool, help='using consistency loss term')
parser.add_argument('--isLP', default=True, type=bool, help='using label propagation')

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml 
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
################################

def main():
    global opts
    opts = parser.parse_args()
    opts.cuda = 0
    print(opts)
    # Set GPU
    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        opts.cuda = 1
        print("Number of device: {}".format(torch.cuda.device_count()))
        print("Currently using GPU {}".format(opts.gpu_ids))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")


    # Set models
    model = ResidualAttentionModel_92_2()


    ch = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, NUM_CLASSES))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))


    ### GPU Setup ###
    if use_gpu:
        model.cuda()
    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model)
        if opts.pause:
            nsml.paused(scope=locals())
    ################################
    if opts.mode == 'train':
        # Define transformations
        weakTransform = transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.RandomResizedCrop(opts.imsize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        StrongTransform = transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.RandomResizedCrop(opts.imsize),
                                  transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        validTransform = transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.CenterCrop(opts.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


        # Set dataloader
        train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
        print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))

        label_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                              transform=weakTransform),
                                batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        print('train_loader done')

        unlabel_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                              transform=strongTransform),
                                batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        print('unlabel_loader done')

        unlabel_unsym_loader = torch.utils.data.DataLoader(     #Apply differenct augmentations to original images
            SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids, unsym=True,
                              transform=strongTransform,
                              transform_base=weakTransform),
                                batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        print('unlabel_unsym_loader done')  

        validation_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                               transform=validTransform),
                               batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        print('validation_loader done')

        model.fc = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, ch))
        bind_nsml(model)
        # model.load(session="kaist007/fashion_dataset/", checkpoint=) # load pretrained model
        
        # Change layers for finetuning
        for param in model.parameters():
            param.requires_grad = False
        
        model.fc = nn.Sequential(model.fc[0], nn.ReLU(), nn.Linear(ch, NUM_CLASSES))
        model = torch.nn.DataParallel(model)
        bind_nsml(model)
        model.to(device)

        # Set hyperparameters for finetuning
        args = loadFineTuningArgs(opts)

        ##### Finetuning-Stage 1: Classification using only labeled data #####

        optimizer = optim.Adam(model.parameters(), lr=args.lr_stg1)
        train_criterion = nn.CrossEntropyLoss()

        bestAcc = -1
        model.train()
        print('start training')
        for epoch in range(1, 1 + args.epoch_stg1):
            fineTuning_base(label_loader, model, train_criterion, optimizer)
            
            acc_top1, acc_top5 = validation(opts, validation_loader, model, epoch, use_gpu)
            is_best = acc_top1 > bestAcc
            bestAcc = max(acc_top1, bestAcc)
            print("epoch {} loss: {}".format(epoch, total_loss))
            nsml.report(summary=True, step=epoch, loss=total_loss.item(), accuracy_1=acc_top1, accuracy_5=acc_top5)
            if is_best:
                print('saving best checkpoint...')
                if IS_ON_NSML:
                    nsml.save(opts.name + '_best')
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_best'))
            if (epoch) % 1 == 0:
                if IS_ON_NSML:
                    nsml.save(opts.name + '_e{}'.format(epoch))
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_e{}'.format(epoch)))

        if not opts.isLP:
            ##### Finetuning-Stage 2: consistency regularization with UDA #####

            assert (opts.isUDA)    

            optimizer = optim.Adam(model.parameters(), lr=args.lr_stg2)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 12, 18, 24], gamma=0.1)
            train_criterion = nn.CrossEntropyLoss()

            bestAcc = -1
            acc_top1, acc_top5 = validation(opts, validation_loader, model, 0, use_gpu)
            print("Starting Accuracy| top-1: {}, top-5: {}".format(prec1, prec5))
            model.train()
            print('start training')
            label_iter = iter(label_loader)
            unlabel_iter = iter(unlabel_unsym_loader)
            for epoch in range(1, 1 + args.epoch_stg2 + args.epoch_stg3):
                fineTuning_UDA(label_loader, unlabel_unsym_loader, model, optimizer, scheduler)

                acc_top1, acc_top5 = validation(opts, validation_loader, model, epoch, use_gpu)
                is_best = acc_top1 > bestAcc
                bestAcc = max(acc_top1, bestAcc)
                nsml.report(summary=True, step=epoch, accuracy_1=acc_top1, accuracy_5=acc_top5)
                if is_best:
                    print('saving best checkpoint...')
                    if IS_ON_NSML:
                        nsml.save(opts.name + '_best')
                    else:
                        torch.save(model.state_dict(), os.path.join('runs', opts.name + '_best'))
                if (epoch) % 1 == 0:
                    if IS_ON_NSML:
                        nsml.save(opts.name + '_e{}'.format(epoch))
                    else:
                        torch.save(model.state_dict(), os.path.join('runs', opts.name + '_e{}'.format(epoch)))

        else:
            ##### Finetuning Stage2: Label propagation occurs and unlabeled data is included for task #####

            for param in model.parameters():
                param.requires_grad = True
                
            # Prepare dataloader for label propagation
            LPLoader, LPLoaderNoshuff, LPData = createTrainLoader(weakTransform, 
                                      validTransform,
                                      DATASET_PATH,
                                      (train_ids, unl_ids),
                                      args,
                                      uda=False,
                                      uda_transformation=StrongTransform)

            # Starting Accuracy
            prec1, prec5 = validation(opts, validation_loader, model, -1, use_gpu)
            print("Starting Accuracy| top-1: {}, top-5: {}".format(prec1, prec5))

            model.train()
            optimizer = torch.optim.SGD(model.parameters(), args.lr_stg2,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay) 
            bestAcc = -1

            # Extracts features and conduct label propagation
            print('Start training')
            print('Extracting features...')
            feats = extractFeatures(LPLoaderNoshuff, model)
            LPData.updatePLabels(feats, k = args.dfs_k, max_iter = 20)

            for epoch in range(1, 1 + args.lr_stg2):
                fineTuning_LP(LPLoader, model, optimizer, epoch, uda=True)
                acc_top1, acc_top5 = validation(opts, validation_loader, model, epoch, use_gpu)
                is_best = acc_top1 > bestAcc
                bestAcc = max(acc_top1, bestAcc)
                if IS_ON_NSML: 
                    nsml.report(summary=True, step=epoch, accuracy_1=acc_top1, accuracy_5=acc_top5)
                    nsml.save(opts.name+'_e{}'.format(epoch))
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name+'_e{}'.format(epoch)))
                if is_best:
                    print('saving best checkpoint...')
                    if IS_ON_NSML:
                        nsml.save(opts.name + '_best')
                    else:
                        torch.save(model.state_dict(), os.path.join('runs', opts.name + '_best'))



            ##### Finetuning Stg3: reduced learning rate, addition of conssistency loss #####

            LPLoader, LPLoaderNoshuff, LPData = createLPTrainLoader(weakTransform, 
                                      validTransform,
                                      DATASET_PATH,
                                      (train_ids, unl_ids),
                                      args,
                                      uda=opts.isUDA,
                                      uda_transformation=StrongTransform)

            # Starting Accuracy
            prec1, prec5 = validation(opts, validation_loader, model, -1, use_gpu)
            print("Starting Accuracy| top-1: {}, top-5: {}".format(prec1, prec5))
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), args.lr_stg3,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay) 
            bestAcc = -1

            # Extracts features and conduct label propagation
            print('Start training')
            print('Extracting features...')
            feats = extractFeatures(LPLoaderNoshuff, model)
            LPData.updatePLabels(feats, k = args.dfs_k, max_iter = 20)

            for epoch in range(1, 1 + args.lr_stg3):
                fineTuning_LP(LPLoader, model, optimizer, epoch, uda=True)
                acc_top1, acc_top5 = validation(opts, validation_loader, model, epoch, use_gpu)
                is_best = acc_top1 > bestAcc
                bestAcc = max(acc_top1, bestAcc)
                if IS_ON_NSML: 
                    nsml.report(summary=True, step=epoch, accuracy_1=acc_top1, accuracy_5=acc_top5)
                    nsml.save(opts.name+'_e{}'.format(epoch))
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name+'_e{}'.format(epoch)))
                if is_best:
                    print('saving best checkpoint...')
                    if IS_ON_NSML:
                        nsml.save(opts.name + '_best')
                    else:
                        torch.save(model.state_dict(), os.path.join('runs', opts.name + '_best'))

# Basic classification task
def fineTuning_base(train_loader, model, train_criterion, optimizer):
    for it, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = data[0].to(device), data[1].to(device)
        p, _ = model(x)
        p = p.div(0.4)
        loss = train_criterion(p, y)
        loss.backward()
        optimizer.step()

# Classification task using Unsuspervised data augmentation 
def fineTuning_UDA(label_loader, unlabel_unsym_loader, model, optimizer, scheduler):
    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabel_unsym_loader)
    for _ in range(15):
        optimizer.zero_grad()
        # Two types of loss from labeled and unlabeled, repectively
        loss_label = 0
        loss_sharp = 0
        # Labeled data
        for _ in range(1):
            try:
                data = next(label_iter)
            except:
                label_iter = iter(label_loader)
                data = next(label_iter)
            x, y = data[0].to(device), data[1].to(device)
            p, _ = model(x)
            p = p.div(0.4)
            loss_label += nn.CrossEntropyLoss()(p, y)
        print("label_loss: {}".format(loss_label))

        # Unlabeled data
        for _ in range(30):
            try:
                data = next(unlabel_iter)
            except:
                unlabel_iter = iter(unlabel_unsym_loader)
                data = next(unlabel_iter)

            # Calculate consistency loss
            d = data.size()
            x = data.view(d[0]*2, d[2], d[3], d[4]).to(device)
            output, _ = model(x)
            output = output.div(0.4)
            p = torch.nn.Softmax(dim=1)(output)
            p1, p2 = p[::2], p[1::2]
            p1_max = torch.max(p1, dim=1).values
            ths = 0.6       #Threshold for confidence
            indice_mask = torch.nonzero(p1_max < ths, as_tuple=True)
            p1[indice_mask] = 0
            loss_sharp -= torch.mean(torch.sum(torch.log(p2.pow(p1)), 1)).item()

        loss_sharp /= 3
        print("loss_sharp: {}".format(loss_sharp))
        loss = loss_label + 2 * loss_sharp
        loss.backward()
        optimizer.step()

    scheduler.step()

# classifiaction task after label propagation
def fineTuning_LP(train_loader, model, optimizer, epoch, uda=False):

    class_criterion = nn.CrossEntropyLoss( ignore_index=NO_LABEL, reduction='none').cuda()
    model.train()
    for i, (batch_input, target, weight, c_weight) in enumerate(train_loader):
        if uda:
            d = batch_input.size()
            input = batch_input.view(d[0]*2, d[2], d[3], d[4])
        else:
            input = batch_input
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target.cuda(async=True))
        weight_var = torch.autograd.Variable(weight.cuda(async=True))
        c_weight_var = torch.autograd.Variable(c_weight.cuda(async=True))
        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        output, _ = model(input_var)
        loss = 0

        #  Calculate consistency loss
        if uda:
            output = output.div(0.4)
            p = torch.nn.Softmax(dim=1)(output)
            p1, p2 = p[::2], p[1::2]
            p1 = p1.detach()
            p1_max = torch.max(p1, dim=1).values
            ths = 0.6
            indice_mask = torch.nonzero(p1_max < ths, as_tuple=True)
            p1[indice_mask] = 0
            loss -= 8 * torch.mean(torch.sum(torch.log(p2.pow(p1)), 1)).item()
            output = output[::2]
    
        loss += class_criterion(output, target_var) 
        loss = loss * weight_var.float()
        loss = loss * c_weight_var
        loss = loss.sum() / minibatch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                'Epoch: [{0}][{1}/{2}]'.format(
                    epoch, i, len(train_loader)))

    return

def validation(opts, validation_loader, model, epoch, use_gpu):
    model.eval()
    avg_top1= 0.0
    avg_top5 = 0.0
    nCnt =0 
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            inputs, labels = data
            if use_gpu :
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            nCnt +=1
            preds, _ = model(inputs)

            acc_top1 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=1)*100
            acc_top5 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=5)*100
            avg_top1 += acc_top1
            avg_top5 += acc_top5

        avg_top1 = float(avg_top1/nCnt)   
        avg_top5= float(avg_top5/nCnt)   
        print('Test Epoch:{} Top1_acc_val:{:.2f}% Top5_acc_val:{:.2f}% '.format(epoch, avg_top1, avg_top5))
    return avg_top1, avg_top5



if __name__ == '__main__':
    main()

