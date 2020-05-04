import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim
import numpy as np
from dataLoaders import create_ensemble_loaders
from utils import build_model, json_file_to_pyobj
import argparse
import os
import random
from tqdm import tqdm
import wandb
import ipdb

abs_path = '/Midgard/home/ferles/Dermatology/src/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def train(args):

    json_options = json_file_to_pyobj(args.config)
    training_configurations = json_options.training
    wandb.init(name=training_configurations.checkpoint+'Ensemble')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pickle_files = [training_configurations.train_pickle, training_configurations.test_pickle]
    train_ind_loaders, train_ood_loaders, test_ind_loaders, test_ood_loaders = create_ensemble_loaders(pickle_files=pickle_files)

    epochs = 90
    criterion = nn.CrossEntropyLoss()
    b = 0.2
    m = 0.4

    model = torch.nn.DataParallel(build_model(args)).to(device) if torch.cuda.device_count() > 1 else build_model(args).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1.25e-02, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

    best_test_acc = 0
    best_test_set_loss = 100

    for index in range(len(train_ind_loaders)):

        train_ind_loader, train_ood_loader = train_ind_loaders[index], train_ood_loaders[index]
        test_ind_loader, test_ood_loader = test_ind_loaders[index], test_ood_loaders[index]

        ood_loader_iter = iter(train_ood_loader)
        ood_test_loader_iter = iter(test_ood_loader)

        for epoch in tqdm(range(epochs)):

            model.train()
            correct, total = 0, 0
            train_loss = 0
            for data in tqdm(train_ind_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                ce_loss = criterion(outputs, labels)

                try:
                    ood_inputs, _ = next(ood_loader_iter)
                except:
                    ood_loader_iter = iter(train_ood_loader)
                    ood_inputs, _ = next(ood_loader_iter)

                ood_inputs = ood_inputs.to(device)
                ood_outputs = model(ood_inputs)
                entropy_input = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * F.softmax(outputs, dim=1), dim=1))
                entropy_output = -torch.mean(torch.sum(F.log_softmax(ood_outputs, dim=1) * F.softmax(ood_outputs, dim=1), dim=1))
                margin_loss = b*torch.clamp(m + entropy_input - entropy_output, min=0)

                loss = ce_loss + margin_loss
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            train_accuracy = correct / total
            wandb.log({'epoch': epoch}, commit=False)
            epoch_train_set_loss = train_loss / train_ind_loader.__len__()

            wandb.log({f'Train Set Loss {index}': epoch_train_set_loss, 'epoch': epoch})
            wandb.log({f'Train Set Accuracy {index}': train_accuracy, 'epoch': epoch})

            model.eval()
            correct, total = 0, 0
            test_loss = 0

            with torch.no_grad():

                for data in test_ind_loader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    t_loss = criterion(outputs, labels)

                    try:
                        ood_inputs, _ = next(ood_test_loader_iter)
                    except:
                        ood_test_loader_iter = iter(test_ood_loader)
                        ood_inputs, _ = next(ood_test_loader_iter)

                    ood_inputs = ood_inputs.to(device)
                    ood_outputs = model(ood_inputs)
                    entropy_input = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * F.softmax(outputs, dim=1), dim=1))
                    entropy_output = -torch.mean(torch.sum(F.log_softmax(ood_outputs, dim=1) * F.softmax(ood_outputs, dim=1), dim=1))
                    margin_loss = b*torch.clamp(m + entropy_input - entropy_output, min=0)

                    test_loss += (t_loss.item() + margin_loss.item())
                    test_loss += t_loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                epoch_accuracy = correct / total

                epoch_test_set_loss = test_loss / test_ind_loader.__len__()
                wandb.log({f'Test Set Loss {index}': epoch_test_set_loss, 'epoch': epoch})
                wandb.log({f'Test Set Accuracy {index}': epoch_accuracy, 'epoch': epoch})

            if epoch_accuracy > best_test_acc:
                best_test_acc = epoch_accuracy
                torch.save(model.state_dict(), f'checkpoints/{training_configurations.checkpoint}_best_accuracy_ensemble_{index}_sees1.pth')

            if epoch_train_set_loss < best_test_set_loss:
                best_test_set_loss = epoch_test_set_loss
                torch.save(model.state_dict(), f'checkpoints/{training_configurations.checkpoint}_best_loss_ensemble_{index}_sees1.pth')

        scheduler.step(epoch=epoch)


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    parser = argparse.ArgumentParser(description='DL Dermatology models')

    parser.add_argument('--config', help='Training Configurations', required=True)

    args = parser.parse_args()
    train(args)

