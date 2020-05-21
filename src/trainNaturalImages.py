import torch
from torch import nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim
from dataLoaders import natural_image_loaders
from utils import build_model, json_file_to_pyobj
import wandb
import argparse
import os
from tqdm import tqdm
import random
import ipdb

abs_path = '/home/ferles/medusa/src/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def train(args):

    json_options = json_file_to_pyobj(args.config)
    training_configurations = json_options.training
    wandb.init(name=training_configurations.checkpoint)
    device = torch.device(f'cuda:{args.device}')

    flag = False
    if training_configurations.train_pickle != 'None' and training_configurations.test_pickle != 'None':
        pickle_files = [training_configurations.train_pickle, training_configurations.test_pickle]
        flag = True

    model = build_model(args).to(device)

    epochs = 40
    dataset = args.dataset.lower()

    if not flag:
        trainloader, val_loader, testloader = natural_image_loaders(dataset, train_batch_size=32, test_batch_size=32, validation_test_split=1000, save_to_pickle=True)
    else:
        trainloader, val_loader, testloader = natural_image_loaders(dataset, train_batch_size=32, test_batch_size=32, validation_test_split=1000, pickle_files=pickle_files)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1.25e-2, momentum=0.9, nesterov=True, weight_decay=1e-4)
    best_test_acc = 0
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    for epoch in tqdm(range(epochs)):

        model.train()
        correct, total = 0, 0
        train_loss = 0
        for data in tqdm(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if 'genOdin' in training_configurations.checkpoint:
                outputs, h, g = model(inputs)
            else:
                outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_accuracy = correct / total
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'Train Set Loss': train_loss / trainloader.__len__(), 'epoch': epoch})
        wandb.log({'Train Set Accuracy': train_accuracy, 'epoch': epoch})

        model.eval()
        correct, total = 0, 0
        test_loss = 0
        scheduler.step(epoch=epoch)

        with torch.no_grad():

            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                if 'genOdin' in training_configurations.checkpoint:
                    outputs, h, g = model(images)
                else:
                    outputs = model(images)
                t_loss = criterion(outputs, labels)
                test_loss += t_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_accuracy = correct / total

            wandb.log({'Test Set Loss': test_loss / testloader.__len__(), 'epoch': epoch})
            wandb.log({'Test Set Accuracy': epoch_accuracy, 'epoch': epoch})

        if epoch_accuracy > best_test_acc:
            best_test_acc = epoch_accuracy
            torch.save(model.state_dict(), f'/raid/ferles/checkpoints/eb0/{training_configurations.checkpoint}_epoch_{epoch}_accuracy_{best_test_acc}.pth')


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    parser = argparse.ArgumentParser(description='DL Dermatology models')

    parser.add_argument('--config', help='Training Configurations', required=True)
    parser.add_argument('--dataset', '--ds', default='cifar10', required=False)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)

    args = parser.parse_args()
    train(args)
