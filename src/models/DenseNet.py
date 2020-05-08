import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random

global_seed = 1
random.seed(global_seed)
np.random.seed(global_seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


class CentroidEuclideanDist(nn.Module):
    def __init__(self, feat_dim, num_centers):
        super(CentroidEuclideanDist, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, feat_dim))
        torch.nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))  # The init is the same as nn.Linear

    def forward(self, feat):
        diff = feat.unsqueeze(dim=1) - self.centers.unsqueeze(dim=0)  # Broadcasting operation
        diff.pow_(2)
        dist = diff.sum(dim=-1)
        return dist


class CosineSimilarity(nn.Module):
    def __init__(self, feat_dim, num_centers):
        super(CosineSimilarity, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, feat_dim))
        torch.nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))  # The init is the same as nn.Linear

    def forward(self, feat):
        feat_normalized = F.normalize(feat)
        center_normalized = F.normalize(self.centers)

        return torch.mm(feat_normalized, center_normalized.t())

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

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate=12, depth=100, reduction=0.5, nClasses=10, bottleneck=True, mode=-1):
        super(DenseNet, self).__init__()

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

        if self.mode == 0:
            self.h = nn.Linear(nChannels, nClasses)
        elif self.mode == 1:
            self.h = CentroidEuclideanDist(nChannels, nClasses)
        else:
            self.h = CosineSimilarity(nChannels, nClasses)

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
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))

        if self.mode == -1:
            return F.log_softmax(self.fc(out))

        else:
            h = self.h(out)
            g = self.g(out)
            g = self.gbn(g)
            g = self.gsigmoid(g)

            if self.mode == 1:
                out = - h / g
            else:
                out = h / g

            return F.log_softmax(out), h, g


def get_optimizer(model):

    weight_decay = 0.0001

    return optim.SGD([
        {'params': model.conv1.parameters(), 'weight_decay':  weight_decay},
        {'params': model.dense1.parameters(), 'weight_decay':  weight_decay},
        {'params': model.trans1.parameters(), 'weight_decay':  weight_decay},
        {'params': model.dense2.parameters(), 'weight_decay':  weight_decay},
        {'params': model.trans2.parameters(), 'weight_decay':  weight_decay},
        {'params': model.dense3.parameters(), 'weight_decay':  weight_decay},
        {'params': model.bn1.parameters(), 'weight_decay':  weight_decay},
        {'params': model.gbn.parameters(), 'weight_decay':  weight_decay},
        {'params': model.g.parameters(), 'weight_decay':  weight_decay},
        {'params': model.h.parameters(), 'weight_decay':  0},
    ], lr=0.1, momentum=0.9)