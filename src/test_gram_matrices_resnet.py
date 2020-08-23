from __future__ import division,print_function
from copy import deepcopy
import argparse
import sys
import time

from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable, grad
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
import calculate_log as callog
import warnings



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.gram_feats = []

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        # torch_model.record(out)
        self.gram_feats.append(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # torch_model.record(out)
        self.gram_feats.append(out)
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.gram_feats = []

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        self.gram_feats.append(out)
        # torch_model.record(out)
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

        self.gram_feats = []

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        # torch_model.record(out)
        self.gram_feats.append(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate=12, depth=100, reduction=0.5, nClasses=10, bottleneck=True, mode=-1):
        super(DenseNet, self).__init__()

        self.collecting = False
        self.gram_feats = []

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)

        self.mode = mode
        if self.mode == -1:
            self.fc = nn.Linear(nChannels, nClasses)
        else:
            self.g = nn.Linear(nChannels, 1)
            self.gbn = nn.BatchNorm1d(1)
            self.gsigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        # self.h = nn.Linear(nChannels, nClasses)
        self.gram_feats = []

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        self.gram_feats.append(out)
        # torch_model.record(out)
        out = self.trans1(out)
        for block in self.trans1.gram_feats:
            self.gram_feats.extend(block.gram_feats)
            block.gram_feats.clear()
        self.gram_feats.append(out)
        out = self.dense1(out)
        for block in self.dense1.gram_feats:
            self.gram_feats.extend(block.gram_feats)
            block.gram_feats.clear()
        self.gram_feats.append(out)
        # torch_model.record(out)
        out = self.trans1(out)
        for block in self.trans2.gram_feats:
            self.gram_feats.extend(block.gram_feats)
            block.gram_feats.clear()
        self.gram_feats.append(out)
        out = self.dense2(out)
        for block in self.dense2.gram_feats:
            self.gram_feats.extend(block.gram_feats)
            block.gram_feats.clear()
        self.gram_feats.append(out)
        # torch_model.record(out)
        out = self.dense3(out)
        for block in self.dense3.gram_feats:
            self.gram_feats.extend(block.gram_feats)
            block.gram_feats.clear()
        self.gram_feats.append(out)
        # torch_model.record(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        self.gram_feats.append(out)
        # torch_model.record(out)
        features = deepcopy(self.gram_feats)
        self.gram_feats.clear()
        return F.log_softmax(self.fc(out)), features

#     def record(self, t):
#         if self.collecting:
#             self.gram_feats.append(t)
#
#     def gram_feature_list(self,x):
#         self.collecting = True
#         self.gram_feats = []
#         self.forward(x)
#         self.collecting = False
#         temp = self.gram_feats
#         self.gram_feats = []
#         return temp
#
#     def get_min_max(self, data, power):
#         mins = []
#         maxs = []
#
#         for i in range(0,len(data),64):
#             batch = data[i:i+64].cuda()
#             feat_list = self.gram_feature_list(batch)
#             for L,feat_L in enumerate(feat_list):
#                 if L==len(mins):
#                     mins.append([None]*len(power))
#                     maxs.append([None]*len(power))
#
#                 for p,P in enumerate(power):
#                     g_p = G_p(feat_L,P)
#
#                     current_min = g_p.min(dim=0,keepdim=True)[0]
#                     current_max = g_p.max(dim=0,keepdim=True)[0]
#
#                     if mins[L][p] is None:
#                         mins[L][p] = current_min
#                         maxs[L][p] = current_max
#                     else:
#                         mins[L][p] = torch.min(current_min,mins[L][p])
#                         maxs[L][p] = torch.max(current_max,maxs[L][p])
#
#         return mins,maxs
#
#     def get_deviations(self,data,power,mins,maxs):
#         deviations = []
#
#         for i in range(0,len(data),64):
#             batch = data[i:i+64].cuda()
#             feat_list = self.gram_feature_list(batch)
#             batch_deviations = []
#             for L,feat_L in enumerate(feat_list):
#                 dev = 0
#                 for p,P in enumerate(power):
#                     g_p = G_p(feat_L,P)
#
#                     dev +=  (F.relu(mins[L][p]-g_p)/torch.abs(mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
#                     dev +=  (F.relu(g_p-maxs[L][p])/torch.abs(maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
#                 batch_deviations.append(dev.cpu().detach().numpy())
#             batch_deviations = np.concatenate(batch_deviations,axis=1)
#             deviations.append(batch_deviations)
#         deviations = np.concatenate(deviations,axis=0)
#
#         return deviations
#
#
# start = time.time()
# parser = argparse.ArgumentParser(description='Out-of-Distribution Detection')
# parser.add_argument('--model_checkpoint', '--mc', required=True)
# parser.add_argument('--device', '--dv', type=int, default=0, required=False)
#
# args = parser.parse_args()
# torch.cuda.set_device(int(args.device))
# device = torch.device(f'cuda:{args.device}')
#
# torch_model = DenseNet()
# state_dict = torch.load(args.model_checkpoint, map_location=device)
# torch_model.load_state_dict(state_dict, strict=False)
# model = torch_model.to(device)
# torch_model.eval()
# print("Loaded DenseNet")
#
# batch_size = 32
# mean = np.array([[125.3/255, 123.0/255, 113.9/255]]).T
#
# std = np.array([[63.0/255, 62.1/255.0, 66.7/255.0]]).T
# normalize = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
#
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize
#
# ])
# transform_test = transforms.Compose([
#     transforms.CenterCrop(size=(32, 32)),
#     transforms.ToTensor(),
#     normalize
# ])
#
# train_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('data', train=True, download=True,
#                      transform=transform_train),
#     batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('data', train=False, transform=transform_test),
#     batch_size=batch_size)
#
#
# data_train = list(torch.utils.data.DataLoader(
#     datasets.CIFAR10('data', train=True, download=True,
#                      transform=transform_test),
#     batch_size=1, shuffle=False))
#
#
# data = list(torch.utils.data.DataLoader(
#     datasets.CIFAR10('data', train=False, download=True,
#                      transform=transform_test),
#     batch_size=1, shuffle=False))
#
# torch_model.eval()
# correct = 0
# total = 0
# for x,y in test_loader:
#     x = x.cuda()
#     y = y.numpy()
#     correct += (y==np.argmax(torch_model(x).detach().cpu().numpy(),axis=1)).sum()
#     total += y.shape[0]
# print("Accuracy: ",correct/total)
#
# cifar100 = list(torch.utils.data.DataLoader(
#     datasets.CIFAR100('data', train=False, download=True,
#                       transform=transform_test),
#     batch_size=1, shuffle=False))
#
# svhn = list(torch.utils.data.DataLoader(
#     datasets.SVHN('data', split="test", download=True,
#                   transform=transform_test),
#     batch_size=1, shuffle=True))
#
# train_preds = []
# train_confs = []
# train_logits = []
# for idx in range(0,len(data_train),batch_size):
#     batch = torch.squeeze(torch.stack([x[0] for x in data_train[idx:idx+batch_size]]),dim=1).cuda()
#
#     logits = torch_model(batch)
#     confs = F.softmax(logits,dim=1).cpu().detach().numpy()
#     preds = np.argmax(confs,axis=1)
#     logits = (logits.cpu().detach().numpy())
#
#     train_confs.extend(np.max(confs,axis=1))
#     train_preds.extend(preds)
#     train_logits.extend(logits)
# print("Train Preds")
#
# test_preds = []
# test_confs = []
# test_logits = []
#
# for idx in range(0,len(data),batch_size):
#     batch = torch.squeeze(torch.stack([x[0] for x in data[idx:idx+batch_size]]),dim=1).cuda()
#
#     logits = torch_model(batch)
#     confs = F.softmax(logits,dim=1).cpu().detach().numpy()
#     preds = np.argmax(confs,axis=1)
#     logits = (logits.cpu().detach().numpy())
#
#     test_confs.extend(np.max(confs,axis=1))
#     test_preds.extend(preds)
#     test_logits.extend(logits)
# print("Test Preds")
#
# import calculate_log as callog
# def detect(all_test_deviations,all_ood_deviations, verbose=True, normalize=True):
#     average_results = {}
#     for i in range(1,11):
#         random.seed(i)
#
#         validation_indices = random.sample(range(len(all_test_deviations)),int(0.1*len(all_test_deviations)))
#         test_indices = sorted(list(set(range(len(all_test_deviations)))-set(validation_indices)))
#
#         validation = all_test_deviations[validation_indices]
#         test_deviations = all_test_deviations[test_indices]
#
#         t95 = validation.mean(axis=0)+10**-7
#         if not normalize:
#             t95 = np.ones_like(t95)
#         test_deviations = (test_deviations/t95[np.newaxis,:]).sum(axis=1)
#         ood_deviations = (all_ood_deviations/t95[np.newaxis,:]).sum(axis=1)
#
#         results = callog.compute_metric(-test_deviations,-ood_deviations)
#         for m in results:
#             average_results[m] = average_results.get(m,0)+results[m]
#
#     for m in average_results:
#         average_results[m] /= i
#     if verbose:
#         callog.print_results(average_results)
#     return average_results
#
#
# def cpu(ob):
#     for i in range(len(ob)):
#         for j in range(len(ob[i])):
#             ob[i][j] = ob[i][j].cpu()
#     return ob
#
# def cuda(ob):
#     for i in range(len(ob)):
#         for j in range(len(ob[i])):
#             ob[i][j] = ob[i][j].cuda()
#     return ob
#
# class Detector:
#     def __init__(self):
#         self.all_test_deviations = None
#         self.mins = {}
#         self.maxs = {}
#         self.classes = range(10)
#
#     def compute_minmaxs(self,data_train,POWERS=[10]):
#         for PRED in tqdm(self.classes):
#             train_indices = np.where(np.array(train_preds)==PRED)[0]
#             train_PRED = torch.squeeze(torch.stack([data_train[i][0] for i in train_indices]),dim=1)
#             mins,maxs = torch_model.get_min_max(train_PRED,power=POWERS)
#             self.mins[PRED] = cpu(mins)
#             self.maxs[PRED] = cpu(maxs)
#             torch.cuda.empty_cache()
#
#     def compute_test_deviations(self,POWERS=[10]):
#         all_test_deviations = None
#         for PRED in tqdm(self.classes):
#             test_indices = np.where(np.array(test_preds)==PRED)[0]
#             test_PRED = torch.squeeze(torch.stack([data[i][0] for i in test_indices]),dim=1)
#             test_confs_PRED = np.array([test_confs[i] for i in test_indices])
#             mins = cuda(self.mins[PRED])
#             maxs = cuda(self.maxs[PRED])
#             test_deviations = torch_model.get_deviations(test_PRED,power=POWERS,mins=mins,maxs=maxs)/test_confs_PRED[:,np.newaxis]
#             cpu(mins)
#             cpu(maxs)
#             if all_test_deviations is None:
#                 all_test_deviations = test_deviations
#             else:
#                 all_test_deviations = np.concatenate([all_test_deviations,test_deviations],axis=0)
#             torch.cuda.empty_cache()
#         self.all_test_deviations = all_test_deviations
#
#     def compute_ood_deviations(self,ood,POWERS=[10]):
#         ood_preds = []
#         ood_confs = []
#
#         for idx in range(0,len(ood),batch_size):
#             batch = torch.squeeze(torch.stack([x[0] for x in ood[idx:idx+batch_size]]),dim=1).cuda()
#             logits = torch_model(batch)
#             confs = F.softmax(logits,dim=1).cpu().detach().numpy()
#             preds = np.argmax(confs,axis=1)
#
#             ood_confs.extend(np.max(confs,axis=1))
#             ood_preds.extend(preds)
#             torch.cuda.empty_cache()
#         print("Done")
#
#         all_ood_deviations = None
#         for PRED in tqdm(self.classes):
#             ood_indices = np.where(np.array(ood_preds)==PRED)[0]
#             if len(ood_indices)==0:
#                 continue
#             ood_PRED = torch.squeeze(torch.stack([ood[i][0] for i in ood_indices]),dim=1)
#             ood_confs_PRED =  np.array([ood_confs[i] for i in ood_indices])
#             mins = cuda(self.mins[PRED])
#             maxs = cuda(self.maxs[PRED])
#             ood_deviations = torch_model.get_deviations(ood_PRED,power=POWERS,mins=mins,maxs=maxs)/ood_confs_PRED[:,np.newaxis]
#             cpu(self.mins[PRED])
#             cpu(self.maxs[PRED])
#             if all_ood_deviations is None:
#                 all_ood_deviations = ood_deviations
#             else:
#                 all_ood_deviations = np.concatenate([all_ood_deviations,ood_deviations],axis=0)
#             torch.cuda.empty_cache()
#         average_results = detect(self.all_test_deviations,all_ood_deviations)
#         return average_results, self.all_test_deviations, all_ood_deviations
#
# def G_p(ob, p):
#     temp = ob.detach()
#
#     temp = temp**p
#     temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
#     temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2)
#     temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)
#
#     return temp
#
# detector = Detector()
# detector.compute_minmaxs(data_train,POWERS=range(1,11))
#
# detector.compute_test_deviations(POWERS=range(1,11))
#
# print("SVHN")
# svhn_results = detector.compute_ood_deviations(svhn,POWERS=range(1,11))
# print("CIFAR-100")
# c100_results = detector.compute_ood_deviations(cifar100,POWERS=range(1,11))
