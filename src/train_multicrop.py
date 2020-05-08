import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from torch.optim.lr_scheduler import StepLR
from utils import build_model, json_file_to_pyobj
from dataLoaders import generate_random_multi_crop_loader
import random
import wandb
from logger import wandb_table
from tqdm import tqdm
import pandas as pd
import ipdb

abs_path = '/home/ferles/medusa/src/'
# global_seed = 1
# torch.backends.cudnn.deterministic = True
# random.seed(global_seed)
# torch.manual_seed(global_seed)
# torch.cuda.manual_seed(global_seed)


def _test_set_eval(net, epoch, device, test_loader, num_classes, columns, gtFile):

    correct, total = 0, 0
    with torch.no_grad():

        net.eval()

        loss_acc = []
        criterion = nn.CrossEntropyLoss()

        paths, results = [], []

        for data in tqdm(test_loader):
            path, images, labels = data
            _, ncrops, c, h , w = images.size()
            images = images.view(ncrops, c, h, w)
            images = images.to(device)
            labels = labels.to(device)

            for i in range(len(path)):
                temp = path[i]
                temp = temp.split('/')[-1].split('.jpg')[0]
                paths.append(temp)

            outputs = net(images)
            softmax_outputs = torch.softmax(outputs, 1)
            outputs = torch.mean(outputs, axis=0)
            softmax_outputs = torch.mean(softmax_outputs, axis=0)
            results.append(softmax_outputs.detach().cpu().numpy().tolist())

            preds = torch.argmax(softmax_outputs)
            _labels = torch.argmax(labels, dim=1)
            correct += ( preds == _labels).sum().item()
            total += _labels.size(0)
            loss = criterion(outputs.unsqueeze(0), _labels)
            loss_acc.append(loss.item())

        accuracy = correct / total
        df = pd.DataFrame(columns=columns)

        for idx, (path, result) in enumerate(zip(paths, results)):
            df.loc[idx] = [path] + result

        df.to_csv(os.path.join(abs_path, 'csvs', f'TemporaryResults-{gtFile}'), index=False)
        os.system(f'isic-challenge-scoring classification {os.path.join(abs_path, "csvs", gtFile)} {os.path.join(abs_path, "csvs", f"TemporaryResults-{gtFile}")} > {os.path.join(abs_path, "txts", gtFile.split(".csv")[0]+"results.txt")}')
        auc, balanced_accuracy = wandb_table(f'{os.path.join(abs_path, "txts", gtFile.split(".csv")[0]+"results.txt")}', epoch, num_classes)

        val_loss = sum(loss_acc) / float(test_loader.__len__())

        wandb.log({'Val Set Loss': val_loss, 'epoch': epoch})
        wandb.log({'Balanced Accuracy': balanced_accuracy, 'epoch': epoch})
        wandb.log({'AUC': auc, 'epoch': epoch})

    return auc, balanced_accuracy, accuracy


def train(args):

    # use_wandb = True
    use_wandb = False

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

    input_size = 224

    train_loader, val_loader, columns = generate_random_multi_crop_loader(csvfiles=[traincsv, testcsv], ncrops=[9, 16], train_batch_size=32, input_size=input_size, gtFile=gtFileName, with_auto_augment=True)

    model = build_model(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = 20
    criterion = nn.CrossEntropyLoss()

    # use_scheduler = True
    use_scheduler = False

    best_auc, best_balanced_accuracy, best_accuracy = 0, 0, 0
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
            # ce_loss = torch.nn.functional.cross_entropy(outputs, _labels, weight=torch.Tensor(frequencies), reduction='none')
            # pt = torch.exp(-ce_loss)
            # focal_loss = (1 * (1-pt)**1 * ce_loss).mean()
            # loss_acc.append(focal_loss.item())
            # focal_loss.backward()
            optimizer.step()
            break

        if use_wandb:
            wandb.log({'Train Set Loss': sum(loss_acc) / float(train_loader.__len__()), 'epoch': epoch})
            wandb.log({'epoch': epoch}, commit=False)
        train_loss.append(sum(loss_acc) / float(train_loader.__len__()))
        loss_acc.clear()

        if use_scheduler:
            scheduler.step()

        auc, balanced_accuracy, accuracy = _test_set_eval(model, epoch, device, val_loader, out_classes, columns, gtFileName)

        if auc > best_auc:
            best_auc = auc
            checkpointFile = os.path.join(f'./checkpoints/isic_classifiers/{checkpointFileName}-best-auc-model-multicrop.pth')
            torch.save(model.state_dict(), checkpointFile)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpointFile = os.path.join(f'./checkpoints/isic_classifiers/{checkpointFileName}-best-accuracy-model-multicrop.pth')
            torch.save(model.state_dict(), checkpointFile)

        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            checkpointFile = os.path.join(f'./checkpoints/isic_classifiers/{checkpointFileName}-best-balanced-accuracy-model-multicrop.pth')
            torch.save(model.state_dict(), checkpointFile)
        else:
            if early_stopping:
                early_stopping_cnt += 1
                if early_stopping_cnt == 3:
                    break


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    parser = argparse.ArgumentParser(description='DL Custom Sets Train')
    parser.add_argument('--config', help='Training Configurations', required=True)
    parser.add_argument('--device', '--dv', help='GPU device', default=0, required=False)
    parser.add_argument('--mode', '--md',  default='new', required=False)

    args = parser.parse_args()
    train(args)


