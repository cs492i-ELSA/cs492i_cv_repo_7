from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torchlars
from residual_attention_network import ResidualAttentionModel_92, ResidualAttentionModel_56
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

import glob

import nsml
from nsml import DATASET_PATH, IS_ON_NSML
# import tqdm
# from tqdm import tqdm_notebook as tqdm
NUM_CLASSES = 265
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
    
def cosine(input, eps=1e-8):
    norm_value = input.norm(p=2, dim=1, keepdim=True)
    return (input@input.t())/(norm_value*norm_value.t()).clamp(min=eps)

def nt_cross_entropy(input, temp=0.5):
    input = cosine(input)
    input = torch.exp(input / temp)
    indices = torch.arange(input.size()[0])
    indices[::2] += 1
    indices[1::2] -= 1
    input = input[indices]
    input = input.diag()/(input.sum(0)-torch.exp(torch.tensor(1/temp)))
    return -torch.log(input.mean())

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):    
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


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
        probs = model(image)
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
parser = argparse.ArgumentParser(description='Team 7')
# basic settings
parser.add_argument('--name',default='nothing', type=str, help='output model name')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=-1, type=int, help='batchsize')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--model', type=str, default='resnet50', help='model')
parser.add_argument('--half', action='store_true', default=False, help='use half-precision')
parser.add_argument('--optimizer', type=str, default='SGD', help="Adam, SGD")
parser.add_argument('--LARS', action='store_true', default=False, help='use LARS')
parser.add_argument('--adaptive_lr', action='store_true', default=False, help='set lr to 0.3*batchsize/256')
parser.add_argument('--scheduler', type=str, default='exp', help='linear, exp')
# basic hyper-parameters
parser.add_argument('--lr', type=float, default=-1, metavar='LR', help='learning rate')
parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml 
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
################################

def main():
    global opts
    opts = parser.parse_args()
    opts.cuda = 0

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

    if opts.model == 'resnet18':
        model = resnet18(pretrained=False)
        if opts.batchsize == -1:
            opts.batchsize = 1024 if opts.half else 512
    elif opts.model == 'resnet50':
        model = resnet50(pretrained=False)
        if opts.batchsize == -1:
            opts.batchsize = 512 if opts.half else 256
    elif opts.model == 'resnet101':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=False)
        if opts.batchsize == -1:
            opts.batchsize = 360 if opts.half else 180
    elif opts.model == 'resnet152':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=False)
        if opts.batchsize == -1:
            opts.batchsize = 256 if opts.half else 128
    elif opts.model == 'ran56':
         model = ResidualAttentionModel_56()
         if opts.batchsize == -1:
            opts.batchsize = 140
    elif opts.model == 'ran92':
         model = ResidualAttentionModel_92()
         if opts.batchsize == -1:
            opts.batchsize = 80      
    

    ch = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, ch))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = torch.nn.DataParallel(model)
    model.eval()
    if opts.half:
        model.half()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

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
    
    bind_nsml(model)
    
    if opts.mode == 'train':
        model.train()
        # Set dataloader
        train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
        print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))
        label_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                              transform=transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.RandomResizedCrop(opts.imsize),
                                  # transforms.Resize((opts.imsize, opts.imsize)),
                                  transforms.RandomHorizontalFlip(),
                                  # transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                batch_size=opts.batchsize * 2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        print('train_loader done')

        unlabel_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                              transform=transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.RandomResizedCrop(opts.imsize),
                                  transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                batch_size=opts.batchsize, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        print('unlabel_loader done')

        validation_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                               transform=transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.CenterCrop(opts.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                               batch_size=opts.batchsize, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        print('validation_loader done')

        model = torch.nn.DataParallel(model) 
        model.to(device)
        bind_nsml(model)
        #Set optimizer
        if opts.optimizer == 'SGD':
            if opts.adaptive_lr:
                base_optimizer = optim.SGD(model.parameters(), lr=0.3*opts.batchsize/256)
            else:
                if opts.lr == -1:
                    base_optimizer = optim.SGD(model.parameters(), lr=0.001)
                else:
                    base_optimizer = optim.SGD(model.parameters(), lr=opts.lr)

        elif opts.optimizer == 'Adam':
            if opts.adaptive_lr:
                base_optimizer = optim.Adam(model.parameters(), lr=0.3*opts.batchsize/256)
            else:
                if opts.lr == -1:
                    base_optimizer = optim.Adam(model.parameters(), lr=0.001)
                else:
                    base_optimizer = optim.Adam(model.parameters(), lr=opts.lr)
        if opts.LARS:
            optimizer = torchlars.LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
        else: 
            optimizer = base_optimizer

        # INSTANTIATE LOSS CLASS
        unlabel_criterion = nt_cross_entropy

        # INSTANTIATE STEP LEARNING SCHEDULER CLASS
        if opts.scheduler == 'linear':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[50, 150], gamma=0.1)
        elif opts.scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-6)

        model.train()
        print('==================================')
        print(opts)
        print('==================================')
        print('starting pretask')
        total_iter = 0
        for epoch in range(1, 201):
            for it, data in enumerate(unlabel_loader):
                total_iter += 1
                d = data.size()
                if opts.half:
                    x = data.view(d[0]*2, d[2], d[3], d[4]).half().to(device)
                else:
                    x = data.view(d[0]*2, d[2], d[3], d[4]).to(device)

                optimizer.zero_grad()
                p = model(x)
                if opts.half:
                    loss = unlabel_criterion(p.float())
                else: 
                    loss = unlabel_criterion(p)

                loss.backward()
                if opts.half:
                    model.float()
                optimizer.step()
                if opts.half:
                    model.half()
                    for layer in model.modules():
                        if isinstance(layer, nn.BatchNorm2d):
                            layer.float()
                print("epoch: ", epoch,  "loss: ", loss.item())
                nsml.report(summary=True, loss=loss.item(), step=total_iter)
            scheduler.step()
            print("epoch: ", epoch,  "loss: ", loss.item())
            if (epoch) % 2 == 0:
                if IS_ON_NSML:
                    nsml.save(opts.name + '_pre{}'.format(epoch))
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_pre{}'.format(epoch))) 

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
            preds = model(inputs)

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

