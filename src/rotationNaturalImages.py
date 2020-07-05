import torch
from torch import nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from dataLoaders import natural_image_loaders
from utils import build_model, build_model_with_checkpoint, json_file_to_pyobj
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
    wandb.init(name=f'rot_{training_configurations.checkpoint}')
    device = torch.device(f'cuda')

    flag = False
    if training_configurations.train_pickle != 'None' and training_configurations.test_pickle != 'None':
        pickle_files = [training_configurations.train_pickle, training_configurations.test_pickle]
        flag = True

    if args.checkpoint is None:
        model = build_model(args, rot=True)
        model = nn.DataParallel(model).to(device)
    else:
        model = build_model_with_checkpoint(modelName='rot' + training_configurations.model.lower(), model_checkpoint=args.checkpoint, device=device, out_classes=training_configurations.out_classes, rot=True)
        model = nn.DataParallel(model).to(device)

    dataset = args.dataset.lower()
    if 'wide' in training_configurations.model.lower():
        resize = False
        epochs = 200
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    else:
        resize = True
        epochs = 40
        optimizer = optim.SGD(model.parameters(), lr=1.25e-2, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    if not flag:
        trainloader, val_loader, testloader = natural_image_loaders(dataset, train_batch_size=32, test_batch_size=16, validation_test_split=1000, save_to_pickle=True, resize=resize)
    else:
        trainloader, val_loader, testloader = natural_image_loaders(dataset, train_batch_size=32, test_batch_size=16, validation_test_split=1000, pickle_files=pickle_files, resize=resize)

    criterion = nn.CrossEntropyLoss()
    checkpoint_val_accuracy, best_val_acc, test_set_accuracy = 0, 0, 0

    train_loss, test_loss = 0, 0
    for epoch in tqdm(range(epochs)):

        model.train()
        correct, total = 0, 0
        for data in tqdm(trainloader):
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

        with torch.no_grad():

            for data in val_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_val_accuracy = correct / total
            wandb.log({'Validation Set Accuracy': epoch_val_accuracy, 'epoch': epoch})

        if epoch_val_accuracy > best_val_acc:
            best_val_acc = epoch_val_accuracy
            if os.path.exists('/raid/ferles/'):
                torch.save(model.state_dict(), f'/raid/ferles/checkpoints/eb0/{dataset}/rot_{training_configurations.checkpoint}.pth')
            else:
                torch.save(model.state_dict(), f'/home/ferles/checkpoints/eb0/{dataset}/rot_{training_configurations.checkpoint}.pth')

            correct, total = 0, 0

            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_set_accuracy = correct / total

        wandb.log({'Test Set Accuracy': test_set_accuracy, 'epoch': epoch})

        scheduler.step(epoch=epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DL Dermatology models')

    parser.add_argument('--config', '--c', help='Training Configurations', required=True)
    parser.add_argument('--dataset', '--ds', default='cifar10', required=False)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)
    parser.add_argument('--checkpoint', '--ck', default=None, required=False)
    args = parser.parse_args()

    visible_divices = f"{args.device}, {args.device+1}"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_divices
    train(args)
