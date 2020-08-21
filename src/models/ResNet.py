import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.gram_feats = []

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        self.gram_feats.append(t)
        self.gram_feats.append(out)
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))
        self.gram_feats.append(t)
        self.gram_feats.append(out)
        t = self.shortcut(x)
        out += t
        self.gram_feats.append(t)
        out = F.relu(out)
        self.gram_feats.append(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.collecting = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        gram_feats = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        for block in self.layer1:
            gram_feats.extend(block.gram_feats)
            block.gram_feats.clear()
        out = self.layer2(out)
        for block in self.layer2:
            gram_feats.extend(block.gram_feats)
            block.gram_feats.clear()
        out = self.layer3(out)
        for block in self.layer3:
            gram_feats.extend(block.gram_feats)
            block.gram_feats.clear()
        out = self.layer4(out)
        for block in self.layer4:
            gram_feats.extend(block.gram_feats)
            block.gram_feats.clear()
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, gram_feats