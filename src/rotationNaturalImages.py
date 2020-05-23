import torch
from torch import nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from dataLoaders import natural_image_loaders
from utils import build_model, json_file_to_pyobj
import argparse
import random
import os
import wandb
from tqdm import tqdm
import ipdb

global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def train(args):

    json_options = json_file_to_pyobj(args.config)
    training_configurations = json_options.training
    wandb.init(name=f'rot{training_configurations.checkpoint}')
    device = torch.device(f'cuda:{args.device}')

    flag = False
    if training_configurations.train_pickle != 'None' and training_configurations.test_pickle != 'None':
        pickle_files = [training_configurations.train_pickle, training_configurations.test_pickle]
        flag = True

    model = nn.DataParallel(build_model(args)).to(device)

    epochs = 90

    if training_configurations.out_classes == 10:
        if not flag:
            trainloader, val_loader, testloader = cifar10loaders(train_batch_size=32, test_batch_size=16, validation_test_split=1000, save_to_pickle=True)
        else:
            trainloader, val_loader, testloader = cifar10loaders(train_batch_size=32, test_batch_size=32, validation_test_split=1000, pickle_files=pickle_files)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1.25e-2, momentum=0.9, nesterov=True, weight_decay=1e-4)

    best_test_acc = 0
    best_test_set_loss = 1e5
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

    train_loss, test_loss = 0, 0
    for epoch in tqdm(range(epochs)):

        model.train()
        correct, total = 0, 0
        for index, data in enumerate(trainloader):
            inputs, labels = data
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            ce_loss = criterion(outputs, labels)
            rot_gt = torch.cat((torch.zeros(inputs.size(0)), torch.ones(inputs.size(0)),
                                2*torch.ones(inputs.size(0)), 3*torch.ones(inputs.size(0))), 0).long().to(device)

            rot_inputs = inputs.detach().cpu().numpy()

            rot_inputs = np.concatenate((rot_inputs, np.rot90(rot_inputs, 1, axes=(2, 3)),
                                         np.rot90(rot_inputs, 2, axes=(2, 3)), np.rot90(rot_inputs, 3, axes=(2, 3))), 0)

            rot_inputs = torch.FloatTensor(rot_inputs)

            rot_preds = model(rot_inputs, rot=True)
            rot_loss = criterion(rot_preds, rot_gt)

            loss = ce_loss + rot_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_accuracy = correct / total
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'Train Set Loss': train_loss / trainloader.__len__(), 'epoch': epoch})
        wandb.log({'Train Set Accuracy': train_accuracy, 'epoch': epoch})

        model.eval()
        correct, total = 0, 0

        scheduler.step(epoch=epoch)
        with torch.no_grad():

            for index, data in enumerate(testloader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                t_ce_loss = criterion(outputs, labels)
                rot_gt = torch.cat((torch.zeros(inputs.size(0)), torch.ones(inputs.size(0)),
                                    2*torch.ones(inputs.size(0)), 3*torch.ones(inputs.size(0))), 0).long().to(device)

                rot_inputs = inputs.detach().cpu().numpy()

                rot_inputs = np.concatenate((rot_inputs, np.rot90(rot_inputs, 1, axes=(2, 3)),
                                             np.rot90(rot_inputs, 2, axes=(2, 3)), np.rot90(rot_inputs, 3, axes=(2, 3))), 0)

                rot_inputs = torch.FloatTensor(rot_inputs)
                if torch.cuda.device_count() == 1:
                    rot_inputs = rot_inputs.to(device)

                rot_preds = model(rot_inputs, rot=True)
                rot_loss = criterion(rot_preds, rot_gt)

                t_loss = t_ce_loss + rot_loss
                test_loss += t_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_accuracy = correct / total
            test_set_loss = test_loss / testloader.__len__()

            wandb.log({'Test Set Loss': test_set_loss, 'epoch': epoch})
            wandb.log({'Test Set Accuracy': epoch_accuracy, 'epoch': epoch})

        if epoch_accuracy > best_test_acc:
            best_test_acc = epoch_accuracy
            torch.save(model.state_dict(), 'checkpoints/eb0Cifar10rotationSeed1_no_loss_momentum.pth')

        if test_set_loss < best_test_set_loss:
            best_test_set_loss = test_set_loss
            torch.save(model.state_dict(), 'checkpoints/eb0Cifar10rotationBestLossSeed1_no_loss_momentum.pth')


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 5"

    parser = argparse.ArgumentParser(description='DL Dermatology models')

    parser.add_argument('--config', help='Training Configurations', required=True)

    args = parser.parse_args()
    train(args)
