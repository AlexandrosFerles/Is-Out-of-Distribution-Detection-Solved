import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from models.DenseNet import DenseNet, get_optimizer
from dataLoaders import cifar10loaders
import os
import argparse
import random
import wandb
from tqdm import tqdm
import ipdb

global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
np.random.seed(global_seed)


def train(mode, dv):

    wandb.init(name=f'DenseNetGenOdinMode={mode}')
    device = torch.device(f'cuda:{dv}')

    model = DenseNet(mode=mode)
    model = model.to(device)
    if mode != -1:
        optimizer = get_optimizer(model)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    pickle_files = ['train_indices_cifar10.pickle', 'val_indices_cifar10.pickle']
    trainloader, _, testloader = cifar10loaders(train_batch_size=64, test_batch_size=64, pickle_files=pickle_files, resize=False)

    epochs = 300
    scheduler = MultiStepLR(optimizer, milestones=[int(0.5*epochs), int(0.75*epochs)], gamma=0.1)

    best_test_acc, best_test_loss = 0, 100

    for epoch in tqdm(range(epochs)):

        model.train()
        correct, total = 0, 0
        train_loss = 0
        for data in tqdm(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if mode == -1:
                outputs = model(inputs)
            else:
                outputs, h, g = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'Train Set Loss': train_loss / trainloader.__len__(), 'epoch': epoch})
        wandb.log({'Train Set Accuracy': train_accuracy, 'epoch': epoch})

        model.eval()
        correct, total = 0, 0
        test_loss = 0

        with torch.no_grad():

            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                if mode == -1:
                    outputs = model(images)
                else:
                    outputs, h, g = model(images)
                t_loss = criterion(outputs, labels)
                test_loss += t_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_test_loss = test_loss / testloader.__len__()
            wandb.log({'Test Set Loss': epoch_test_loss, 'epoch': epoch})
            epoch_accuracy = correct / total
            wandb.log({'Test Set Accuracy': epoch_accuracy, 'epoch': epoch})

        if epoch_accuracy > best_test_acc:
            best_test_acc = epoch_accuracy
            torch.save(model.state_dict(), f'/raid/ferles/checkpoints/dense/dense_net_godin_{mode}_epoch_{epoch}_acc_{best_test_acc}.pth')

        scheduler.step(epoch=epoch)



if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    parser = argparse.ArgumentParser(description='DenseNetGodin')

    parser.add_argument('--mode', type=int, default=-1, required=False)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)

    args = parser.parse_args()
    train(args.mode, args.device)
