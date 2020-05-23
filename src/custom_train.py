import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from utils import build_model
from dataLoaders import oversampling_loaders_custom, oversampling_loaders_exclude_class_custom_no_gts
from utils import json_file_to_pyobj
import random
import wandb
from logger import wandb_table
from tqdm import tqdm
import pandas as pd
import ipdb

abs_path = '/home/ferles/medusa/src/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _test_set_eval(net, epoch, device, test_loader, num_classes, columns, gtFile):

    with torch.no_grad():

        net.eval()

        loss_acc = []
        criterion = nn.CrossEntropyLoss()

        paths, results = [], []

        correct, total = 0, 0

        for data in tqdm(test_loader):
            path, images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            for i in range(len(path)):
                temp = path[i]
                temp = temp.split('/')[-1].split('.jpg')[0]
                paths.append(temp)

            outputs = net(images)
            softmax_outputs = torch.softmax(outputs, 1)
            max_idx = torch.argmax(softmax_outputs, axis=1)
            for output in softmax_outputs:
                temp = output.detach().cpu().numpy().tolist()
                results.append([float(elem) for elem in temp])

            _labels = torch.argmax(labels, dim=1)
            correct += (max_idx == _labels).sum().item()
            total += max_idx.size()[0]
            loss = criterion(outputs, _labels)
            loss_acc.append(loss.item())

        detection_accuracy = round(100*correct/total, 2)

        df = pd.DataFrame(columns=columns)
        
        for idx, (path, result) in enumerate(zip(paths, results)):
            df.loc[idx] = [path] + result

        df.to_csv(os.path.join(abs_path, 'csvs', f'TemporaryResults-{gtFile}'), index=False)
        os.system(f'isic-challenge-scoring classification {os.path.join(abs_path, "csvs", gtFile)} {os.path.join(abs_path, "csvs", f"TemporaryResults-{gtFile}")} > {os.path.join(abs_path, "txts", gtFile.split(".csv")[0]+"results.txt")}')
        auc, balanced_accuracy = wandb_table(f'{os.path.join(abs_path, "txts", gtFile.split(".csv")[0]+"results.txt")}', epoch, num_classes)

        val_loss = sum(loss_acc) / float(test_loader.__len__())

        wandb.log({'Val Set Loss': val_loss, 'epoch': epoch})
        wandb.log({'Balanced Accuracy': balanced_accuracy, 'epoch': epoch})
        wandb.log({'Detection Accuracy': detection_accuracy, 'epoch': epoch})
        wandb.log({'AUC': auc, 'epoch': epoch})

    return balanced_accuracy


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

    mode = args.mode
    if use_wandb:
        wandb.init(name=checkpointFileName)

    if exclude_class is None:
        train_loader, val_loader, test_loader, columns = oversampling_loaders_custom(csvfiles=[traincsv, testcsv], train_batch_size=32, val_batch_size=16, gtFile=gtFileName)
    else:
        train_loader, val_loader, test_loader, columns = oversampling_loaders_exclude_class_custom_no_gts(csvfiles=[traincsv, testcsv], train_batch_size=32, val_batch_size=16, gtFile=gtFileName, exclude_class=exclude_class)

    model = build_model(args).to(device)
    epochs = 40
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1.25e-2, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    best_auc, best_balanced_accuracy = 0, 0
    train_loss, val_loss, balanced_accuracies = [], [], []

    early_stopping = False
    early_stopping_cnt = 0
        
    for epoch in tqdm(range(epochs)):

        model.train()
        loss_acc = []

        for data in tqdm(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            _labels = torch.argmax(labels, dim=1)
            loss = criterion(outputs, _labels)
            loss_acc.append(loss.item())
            loss.backward()
            optimizer.step()

        wandb.log({'Train Set Loss': sum(loss_acc) / float(train_loader.__len__()), 'epoch': epoch})
        wandb.log({'epoch': epoch}, commit=False)
        train_loss.append(sum(loss_acc) / float(train_loader.__len__()))
        loss_acc.clear()

        balanced_accuracy = _test_set_eval(model, epoch, device, val_loader, out_classes, columns, gtFileName)

        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            checkpointFile = os.path.join(f'/raid/ferles/checkpoints/isic_classifiers/{checkpointFileName}-best-balanced-accuracy-model_{mode}_next.pth')
            torch.save(model.state_dict(), checkpointFile)
        else:
            if early_stopping:
                early_stopping_cnt += 1
                if early_stopping_cnt == 3:
                    break

        scheduler.step()


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    parser = argparse.ArgumentParser(description='DL Custom Sets Train')
    parser.add_argument('--config', help='Training Configurations', required=True)
    parser.add_argument('--device', '--dv', help='GPU device', default=0, required=False)
    parser.add_argument('--mode', '--md',  default='new', required=False)

    args = parser.parse_args()
    train(args)


