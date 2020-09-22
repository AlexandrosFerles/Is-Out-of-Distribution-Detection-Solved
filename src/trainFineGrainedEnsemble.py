import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim
from dataLoaders import fine_grained_image_loaders, fine_grained_image_loaders_subset
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
    wandb.init(name=f"{training_configurations.checkpoint}_subset_{args.subset_index}_ensemble")
    device = torch.device(f'cuda:{args.device}')

    flag = False
    if training_configurations.train_pickle != 'None' and training_configurations.test_pickle != 'None':
        pickle_files = [training_configurations.train_pickle, training_configurations.test_pickle]
        flag = True

    if args.subset_index is None:
        model = build_model(args)
        model = model.to(device)
        epochs = 40
        optimizer = optim.SGD(model.parameters(), lr=1.25e-2, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    dataset = args.dataset.lower()
    b = 0.2
    m = 0.4

    if not flag:
        trainloader, val_loader, testloader = fine_grained_image_loaders_subset(dataset, subset_index=args.subset_index, validation_test_split=800, save_to_pickle=True)
    else:
        pickle_files[0] = "pickle_files/"+pickle_files[0].split(".pickle")[0]+f"_subset_{args.subset_index}.pickle"
        pickle_files[1] = "pickle_files/"+pickle_files[1].split(".pickle")[0]+f"_subset_{args.subset_index}.pickle"
        trainloader, val_loader, testloader, num_classes = fine_grained_image_loaders_subset(dataset, subset_index=args.subset_index, validation_test_split=800, pickle_files=pickle_files, ret_num_classes=True)
        train_ood_loader = fine_grained_image_loaders_subset(dataset, single=True, subset_index=args.subset_index, validation_test_split=800, pickle_files=pickle_files)

        if 'genOdin' in training_configurations.checkpoint:
            weight_decay = 1e-4
            optimizer = optim.SGD([
                {'params': model._conv_stem.parameters(), 'weight_decay':  weight_decay},
                {'params': model._bn0.parameters(), 'weight_decay':  weight_decay},
                {'params': model._blocks.parameters(), 'weight_decay':  weight_decay},
                {'params': model._conv_head.parameters(), 'weight_decay':  weight_decay},
                {'params': model._bn1.parameters(), 'weight_decay':  weight_decay},
                {'params': model._fc_denominator.parameters(), 'weight_decay':  weight_decay},
                {'params': model._denominator_batch_norm.parameters(), 'weight_decay':  weight_decay},
                {'params': model._fc_nominator.parameters(), 'weight_decay':  0},
            ], lr=1.25e-2, momentum=0.9, nesterov=True)
            scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    if args.subset_index is not None:
        model = build_model(args)
        model._fc = nn.Linear(model._fc.in_features, num_classes)
        model = model.to(device)
        epochs = 40
        optimizer = optim.SGD(model.parameters(), lr=1.25e-2, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    checkpoint_val_accuracy, best_val_acc, test_set_accuracy = 0, 0, 0

    ood_loader_iter = iter(train_ood_loader)

    for epoch in tqdm(range(epochs)):

        model.train()
        correct, total = 0, 0
        train_loss = 0
        for data in tqdm(trainloader):

            model.train()

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
                torch.save(model.state_dict(), f'/raid/ferles/checkpoints/eb0/{dataset}/{training_configurations.checkpoint}_subset_ens_{args.subset_index}.pth')
            else:
                torch.save(model.state_dict(), f'/home/ferles/checkpoints/eb0/{dataset}/{training_configurations.checkpoint}_subset_ens_{args.subset_index}.pth')

            correct, total = 0, 0

            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                if 'genodin' in training_configurations.checkpoint.lower():
                    outputs, h, g = model(images)
                else:
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_set_accuracy = correct / total

        wandb.log({'Test Set Accuracy': test_set_accuracy, 'epoch': epoch})

        scheduler.step(epoch=epoch)


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    parser = argparse.ArgumentParser(description='DL Dermatology models')

    parser.add_argument('--config', help='Training Configurations', required=True)
    parser.add_argument('--dataset', '--ds', default='stanforddogs', required=False)
    parser.add_argument('--subset_index', '--sub', type=int, required=True)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)

    args = parser.parse_args()
    train(args)
