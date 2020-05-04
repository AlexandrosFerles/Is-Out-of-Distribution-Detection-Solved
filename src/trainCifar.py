import torch
from torch import nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim
from dataLoaders import cifar10loaders, cifar100loaders
import wandb
from utils import build_model, json_file_to_pyobj
import argparse
import os
from tqdm import tqdm
import random
import ipdb

abs_path = '/Midgard/home/ferles/Dermatology/src/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


if __name__=='__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    # TODO: Make them a config file
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataset', '--d', required=True)
    parser.add_argument('--checkpoint', '--cf', required=True)
    parser.add_argument('--mode', '--md', default='standard', required=True)
    parser.add_argument('--model', '--m', default='eb0', required=False)
    parser.add_argument('--num_classes', '--nc', type=int, default=8, required=False)
    parser.add_argument('--train_batch_size', '--bs', type=int, default=32, required=False)
    parser.add_argument('--test_batch_size', '--tbs', type=int, default=10, required=False)
    parser.add_argument('--traincsv', '--tcsv', required=False)
    parser.add_argument('--val_csv', '--vcsv', required=False)
    parser.add_argument('--exclude_class', '--ex', required=False)
