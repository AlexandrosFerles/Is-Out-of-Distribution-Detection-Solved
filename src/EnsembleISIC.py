import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
from torch.optim.lr_scheduler import StepLR
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

    use_wandb = True

    device = torch.device(f'cuda:{args.device}')

    json_options = json_file_to_pyobj(args.config)
    training_configurations = json_options.training
    traincsv = training_configurations.traincsv
    testcsv = training_configurations.testcsv
    gtFileName = training_configurations.gtFile
    checkpointFileName = training_configurations.checkpointFile
    out_classes = training_configurations.out_classes
    exclude_class = training_configurations.exclude_class
    exclude_class = None if exclude_class == "None" else exclude_class

    if use_wandb:
        wandb.init(name=checkpointFileName)

    batch_size = 32

    if exclude_class is None:
        train_loader, val_loader, test_loader, columns = oversampling_loaders_custom(csvfiles=[traincsv, testcsv], train_batch_size=32, val_batch_size=16, gtFile=gtFileName)
    else:
        train_loader, val_loader, test_loader, columns = oversampling_loaders_exclude_class_custom_no_gts(csvfiles=[traincsv, testcsv], train_batch_size=32, val_batch_size=16, gtFile=gtFileName, exclude_class=exclude_class)
    _, _, _, ood_loader = _get_isic_loaders_ood(exclude_class=exclude_class, batch_size=batch_size)
    ood_loader_iter = iter(ood_loader)
    model = build_model(args)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = 50
    criterion = nn.CrossEntropyLoss()
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    use_scheduler = True

    best_auc, best_balanced_accuracy = 0, 0
    train_loss, val_loss, balanced_accuracies = [], [], []
    best_train_loss = 1e30

    b = 0.2
    m = 0.4

    for epoch in tqdm(range(epochs)):

        model.train()
        loss_acc = []

        for data in tqdm(train_loader):
            path, inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            try:
                ood_inputs, _ = next(ood_loader_iter)
            except:
                ood_loader_iter = iter(ood_loader)
                ood_inputs, _ = next(ood_loader_iter)

            ood_inputs = ood_inputs.to(device)
            ood_outputs = model(ood_inputs)

            entropy_input = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * F.softmax(outputs, dim=1), dim=1))
            entropy_output = -torch.mean(torch.sum(F.log_softmax(ood_outputs, dim=1) * F.softmax(ood_outputs, dim=1), dim=1))

            _labels = torch.argmax(labels, dim=1)
            ce_loss = criterion(outputs, _labels)
            margin_loss = b*torch.clamp(m + entropy_input - entropy_output, min=0)

            loss = ce_loss + margin_loss

            loss_acc.append(loss.item())
            loss.backward()
            optimizer.step()

        wandb.log({'Train Set Loss': sum(loss_acc) / float(train_loader.__len__()), 'epoch': epoch})
        wandb.log({'epoch': epoch}, commit=False)
        train_loss.append(sum(loss_acc) / float(train_loader.__len__()))
        loss_acc.clear()
        if use_scheduler:
            scheduler.step()

        if train_loss[-1] < best_train_loss:
            best_train_loss = train_loss[-1]
            checkpointFile = os.path.join(f'/home/ferles/Dermatology/src/checkpoints/{checkpointFileName}-best-ensemble-train-loss-model.pth')
            torch.save(model.state_dict(), checkpointFile)

        auc, balanced_accuracy = _test_set_eval(model, epoch, device, val_loader, out_classes, columns, gtFileName)

        if auc > best_auc:
            best_auc = auc
            checkpointFile = os.path.join(f'/home/ferles/Dermatology/medusa/checkpoints/{checkpointFileName}-best-ensemble-auc-model.pth')
            torch.save(model.state_dict(), checkpointFile)

        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            checkpointFile = os.path.join(f'/home/ferles/Dermatology/medusa/checkpoints/{checkpointFileName}-best-ensemble-balanced-accuracy-model.pth')
            torch.save(model.state_dict(), checkpointFile)


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    parser = argparse.ArgumentParser(description='DL Dermatology models')
    parser.add_argument('--config', help='Training Configurations', required=True)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)
    parser.add_argument('--mode', '--md', type=str, default='new', required=False)

    args = parser.parse_args()
    train(args)
