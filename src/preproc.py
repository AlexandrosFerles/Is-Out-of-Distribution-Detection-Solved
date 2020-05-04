import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
import pandas as pd
import os
import random


class MyImageFolder(ImageFolder):
    def getitem(self, index):
        return super(MyImageFolder, self).getitem(index), self.imgs[index]


def _setup_data(folder,resolution=224):

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor()
    ])

    dataset = MyImageFolder(root=folder, transform=transform)
    loader = DataLoader(dataset, batch_size=4, num_workers=4)

    return loader


def main():

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    checkpointFile = './checkpoint/EB6-auto-aug-step-lr-best-balanced-accuracy-model.pth'

    model = EfficientNet.from_pretrained('efficientnet-b6')

    model._fc = nn.Linear(model._fc.in_features, 7)
    model = model.to(device)

    print('Loading model....')
    state_dict = torch.load(checkpointFile)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key.split('module.')[1]
        new_state_dict[new_key] = value

    torch.save(new_state_dict, checkpointFile.split('.pth')[0]+'correct.pth')
    model.load_state_dict(torch.load(checkpointFile.split('.pth')[0]+'correct.pth', map_location=device))
    os.system(f"rm {checkpointFile.split('.pth')[0]+'correct.pth'}")
    print('Done!')

    paths, results = [], []
    columns = ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    model.eval()

    loader = _setup_data(folder='Preproc')
    results = []
    for paths, images in loader:
        images = images.to(device)

        outputs = model(images)
        softmax_outputs = torch.softmax(outputs, 1)

        for output in softmax_outputs:
            temp = output.detach().cpu().numpy().tolist()
            results.append([float(elem) for elem in temp]+[1e-30])

    df = pd.DataFrame(columns=columns)

    for idx, (path, result) in enumerate(zip(paths, results)):
        df.loc[idx] = [path] + result
    df.to_csv(f'preproc.csv', index=False)
