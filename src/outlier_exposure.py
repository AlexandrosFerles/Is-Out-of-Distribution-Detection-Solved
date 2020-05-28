import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from utils import build_model
from dataLoaders import oversampling_loaders_exclude_class_custom_no_gts, oversampling_loaders_custom, _get_isic_loaders_ood, imageNetLoader
from utils import json_file_to_pyobj
import wandb
from logger import wandb_table
from tqdm import tqdm
import pandas as pd
import random
import ipdb

abs_path = '/home/ferles/Dermatology/medusa/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _test_set_eval(net, epoch, device, test_loader, num_classes, columns, gtFile):

    with torch.no_grad():

        net.eval()

        loss_acc = []
        criterion = nn.CrossEntropyLoss()

        paths, results = [], []

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
            for output in softmax_outputs:
                temp = output.detach().cpu().numpy().tolist()
                results.append([float(elem) for elem in temp])

            _labels = torch.argmax(labels, dim=1)
            ce_loss = criterion(outputs, _labels)

            loss = ce_loss
            loss_acc.append(loss.item())

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

    return auc, balanced_accuracy


def train(args):

    device = torch.device(f'cuda:{args.device}')

    json_options = json_file_to_pyobj(args.config)
    training_configurations = json_options.training
    traincsv = training_configurations.traincsv
    testcsv = training_configurations.testcsv
    gtFileName = training_configurations.gtFile
    out_classes = training_configurations.out_classes
    exclude_class = training_configurations.exclude_class
    exclude_class = None if exclude_class == "None" else exclude_class

    if exclude_class is None:
        wandb.init(name='oe_isic')
    else:
        wandb.init(name=f'oe_{exclude_class}')

    batch_size = 32

    if exclude_class is None:
        train_loader, val_loader, test_loader, columns = oversampling_loaders_custom(csvfiles=[traincsv, testcsv], train_batch_size=32, val_batch_size=16, gtFile=gtFileName)
    else:
        train_loader, val_loader, test_loader, columns = oversampling_loaders_exclude_class_custom_no_gts(csvfiles=[traincsv, testcsv], train_batch_size=32, val_batch_size=16, gtFile=gtFileName, exclude_class=exclude_class)
    ood_loader = imageNetLoader(dataset='isic', batch_size=batch_size)
    ood_loader_iter = iter(ood_loader)

    model = build_model(args).to(device)
    epochs = 40
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1.25e-2, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    uniform = torch.ones(size=(batch_size, out_classes)) / float(out_classes)
    uniform = uniform.to(device)
    lamda = 0.5

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

            try:
                ood_inputs, _ = next(ood_loader_iter)
            except:
                ood_loader_iter = iter(ood_loader)
                ood_inputs, _ = next(ood_loader_iter)

            ood_inputs = ood_inputs.to(device)
            ood_outputs = model(ood_inputs)

            _labels = torch.argmax(labels, dim=1)
            ce_loss = criterion(outputs, _labels)
            if ood_outputs.size(0) < batch_size:
                uniform = torch.ones(size=(ood_outputs.size(0), out_classes)) / float(out_classes)
                uniform = uniform.to(device)

            outlier_loss = lamda * -(uniform.mean(1) - torch.logsumexp(ood_outputs, dim=1)).mean()
            loss = ce_loss + outlier_loss

            loss_acc.append(loss.item())
            loss.backward()
            optimizer.step()

            if ood_outputs.size(0) < batch_size:
                uniform = torch.ones(size=(batch_size, out_classes)) / float(out_classes)
                uniform = uniform.to(device)
            break

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
                    if exclude_class is None:
                        torch.save(model.state_dict(), f'/raid/ferles/checkpoints/isic_classifiers/outlier_exposure_isic.pth')
                    else:
                        torch.save(model.state_dict(), f'/raid/ferles/checkpoints/isic_classifiers/outlier_exposure_{exclude_class}.pth')
                else:
                    if exclude_class is None:
                        torch.save(model.state_dict(), f'/home/ferles/checkpoints/isic_classifiers/outlier_exposure_isic.pth')
                    else:
                        torch.save(model.state_dict(), f'/home/ferles/checkpoints/isic_classifiers/outlier_exposure_{exclude_class}.pth')
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

    parser = argparse.ArgumentParser(description='DL Custom Sets Train')
    parser.add_argument('--config', help='Training Configurations', required=True)
    parser.add_argument('--device', '--dv', help='GPU device', default=0, required=False)
    parser.add_argument('--mode', '--md',  default='new', required=False)

    args = parser.parse_args()
    train(args)
