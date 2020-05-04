import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from utils import build_model_with_checkpoint
from dataLoaders import _get_7point_loaders, _get_isic_loader
from utils import json_file_to_pyobj
import random
import wandb
from logger import wandb_table
from tqdm import tqdm
import pandas as pd
import ipdb

abs_path = '/home/ferles/Dermatology/medusa/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def train(args):

    device = f"cuda:{args.device}"
    isic_loader = _get_isic_loader(batch_size=20)
    loader, columns = _get_7point_loaders(batch_size=20)
    model = build_model_with_checkpoint('eb0', args.model_checkpoint, device=device, out_classes=args.num_classes)

    with torch.no_grad():

        model.eval()
        paths, results = [], []
        preds_acc, gts_acc = np.zeros(0), np.zeros(0)

        for data in tqdm(isic_loader):

            path, images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            for i in range(len(path)):
                temp = path[i]
                temp = temp.split('/')[-1].split('.jpg')[0]
                paths.append(temp)

            outputs = model(images)
            softmax_outputs = torch.softmax(outputs, 1)
            max_idx = torch.argmax(softmax_outputs)
            for output in softmax_outputs:
                temp = output.detach().cpu().numpy().tolist()
                results.append([float(elem) for elem in temp])

            _labels = torch.argmax(labels, dim=1)
            preds_acc = np.append(preds_acc, max_idx.detach().cpu().numpy())
            gts_acc = np.append(gts_acc, _labels.detach().cpu().numpy())

        df = pd.DataFrame(columns=columns)
        for idx, (path, result) in enumerate(zip(paths, results)):
            df.loc[idx] = [path] + result

        df.to_csv(os.path.join(abs_path, 'temp_7point.csv'), index=False)
        os.system(f'isic-challenge-scoring classification /raid/ferles/7-point/7pointAsISIC.csv temp_7point.csv')



if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    parser = argparse.ArgumentParser(description='DL Custom Sets Train')
    parser.add_argument('--model_checkpoint', '--mc', required=True)
    parser.add_argument('--csv_file', '--csv',  required=False)
    parser.add_argument('--num_classes', '--nc', default=8, required=False)
    parser.add_argument('--device', '--dv', help='GPU device', default=0, required=False)

    args = parser.parse_args()
    train(args)
