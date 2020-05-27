import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from utils import build_model
from dataLoaders import oversampling_loaders_exclude_class_custom_no_gts, oversampling_loaders_custom, _get_isic_loaders_ood
from utils import json_file_to_pyobj
import wandb
from logger import wandb_table
from tqdm import tqdm
import pandas as pd
import ipdb

abs_path = '/home/ferles/medusa/src'


def _test_set_eval(net, epoch, device, test_loader, num_classes, columns, gtFile):

    with torch.no_grad():

        net.eval()

        loss_acc = []
        criterion = nn.CrossEntropyLoss()

        correct, total = 0, 0
        for data in tqdm(test_loader):
            path, images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            softmax_outputs = torch.softmax(outputs, 1)
            max_idx = torch.argmax(softmax_outputs, axis=1)

            _labels = torch.argmax(labels, dim=1)
            correct += (max_idx == _labels).sum().item()
            total += max_idx.size()[0]
            loss = criterion(outputs, _labels)
            loss_acc.append(loss.item())

        detection_accuracy = round(100*correct/total, 2)
        val_loss = sum(loss_acc) / float(test_loader.__len__())

    return val_loss, detection_accuracy


def train(args):

    device = torch.device(f'cuda:{args.device}')
    json_options = json_file_to_pyobj(args.config)
    training_configurations = json_options.training
    traincsv = training_configurations.traincsv
    testcsv = training_configurations.testcsv
    gtFileName = training_configurations.gtFile
    checkpointFileName = training_configurations.checkpointFile
    exclude_class = training_configurations.exclude_class

    wandb.init(name=checkpointFileName)

    batch_size = 32

    if exclude_class is None:
        train_loader, val_loader, test_loader, columns = oversampling_loaders_custom(csvfiles=[traincsv, testcsv], train_batch_size=32, val_batch_size=16, gtFile=gtFileName)
    else:
        train_loader, val_loader, test_loader, columns = oversampling_loaders_exclude_class_custom_no_gts(csvfiles=[traincsv, testcsv], train_batch_size=32, val_batch_size=16, gtFile=gtFileName, exclude_class=exclude_class)
    _, _, _, ood_loader = _get_isic_loaders_ood(exclude_class=exclude_class, batch_size=batch_size)

    ood_loader_iter = iter(ood_loader)

    model = build_model(args).to(device)
    epochs = 40
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1.25e-2, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    checkpoint_val_accuracy, best_val_acc, test_set_accuracy = 0, 0, 0

    b = 0.2
    m = 0.4

    for epoch in tqdm(range(epochs)):

        model.train()
        loss_acc = []

        for data in tqdm(train_loader):

            model.train()

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _labels = torch.argmax(labels, dim=1)
            ce_loss = criterion(outputs, _labels)

            try:
                ood_inputs, _ = next(ood_loader_iter)
            except:
                ood_loader_iter = iter(ood_loader)
                ood_inputs, _ = next(ood_loader_iter)

            ood_inputs = ood_inputs.to(device)
            ood_outputs = model(ood_inputs)
            entropy_input = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * F.softmax(outputs, dim=1), dim=1))
            entropy_output = -torch.mean(torch.sum(F.log_softmax(ood_outputs, dim=1) * F.softmax(ood_outputs, dim=1), dim=1))
            margin_loss = b*torch.clamp(m + entropy_input - entropy_output, min=0)

            loss = ce_loss + margin_loss
            loss_acc.append(loss.item())
            loss.backward()
            optimizer.step()

        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'Train Set Loss': sum(loss_acc) / float(train_loader.__len__()), 'epoch': epoch})

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():

            for data in val_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                _labels = torch.argmax(labels, dim=1)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == _labels).sum().item()

            val_detection_accuracy = round(100*correct/total, 2)
            wandb.log({'Validation Detection Accuracy': val_detection_accuracy, 'epoch': epoch})

            if val_detection_accuracy > best_val_acc:
                best_val_acc = val_detection_accuracy

                if os.path.exists('/raid/ferles/'):
                    torch.save(model.state_dict(), f'/raid/ferles/checkpoints/isic_classifiers/{training_configurations.checkpointFile}.pth')
                else:
                    torch.save(model.state_dict(), f'/home/ferles/checkpoints/isic_classifiers/{training_configurations.checkpointFile}.pth')

                correct, total = 0, 0

                for data in test_loader:
                    _, images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    _labels = torch.argmax(labels, dim=1)

                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == _labels).sum().item()

                test_detection_accuracy = correct / total

            wandb.log({'Detection Accuracy': test_detection_accuracy, 'epoch': epoch})

            scheduler.step(epoch=epoch)


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    parser = argparse.ArgumentParser(description='DL Dermatology models')
    parser.add_argument('--config', help='Training Configurations', required=True)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)

    args = parser.parse_args()
    train(args)
