import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from utils import build_model
from dataLoaders import oversampling_loaders_exclude_class_custom_no_gts, oversampling_loaders_custom
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
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _test_set_eval(net, epoch, device, test_loader, num_classes, columns, gtFile, fold_index=None):

    with torch.no_grad():

        net.eval()

        loss_acc = []
        criterion = nn.CrossEntropyLoss()

        # paths, results = [], []
        correct, total = 0, 0

        for data in tqdm(test_loader):
            path, images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # for i in range(len(path)):
            #     temp = path[i]
            #     temp = temp.split('/')[-1].split('.jpg')[0]
            #     paths.append(temp)

            outputs = net(images)
            softmax_outputs = torch.softmax(outputs, 1)
            max_idx = torch.argmax(softmax_outputs, axis=1)
            # for output in softmax_outputs:
            #     temp = output.detach().cpu().numpy().tolist()
            #     results.append([float(elem) for elem in temp])

            _labels = torch.argmax(labels, dim=1)
            correct += (max_idx == _labels).sum().item()
            total += max_idx.size()[0]
            loss = criterion(outputs, _labels)
            loss_acc.append(loss.item())
            # ce_loss = criterion(outputs, _labels)
            # rot_gt = torch.cat((torch.zeros(images.size(0)), torch.ones(images.size(0)),
            #                     2*torch.ones(images.size(0)), 3*torch.ones(images.size(0))), 0).long().to(device)
            # rot_inputs = images.cpu().numpy()
            #
            # rot_inputs = np.concatenate((rot_inputs, np.rot90(rot_inputs, 1, axes=(2, 3)),
            #                              np.rot90(rot_inputs, 2, axes=(2, 3)), np.rot90(rot_inputs, 3, axes=(2, 3))), 0)
            # rot_inputs = torch.FloatTensor(rot_inputs)
            # rot_preds = net(rot_inputs, rot=True)
            #
            # rot_loss = 0.5 * criterion(rot_preds, rot_gt)
            # loss = ce_loss + rot_loss
            #
            # loss_acc.append(loss.item())

        # df = pd.DataFrame(columns=columns)
        # for idx, (path, result) in enumerate(zip(paths, results)):
        #     df.loc[idx] = [path] + result
        # if fold_index is not None:
        #     gtFile = gtFile.split('.csv')[0]+f'fold{fold_index}'+'.csv'
        #
        # for idx, (path, result) in enumerate(zip(paths, results)):
        #     df.loc[idx] = [path] + result
        #
        # df.to_csv(os.path.join(abs_path, 'csvs', f'TemporaryResults-{gtFile}'), index=False)
        # os.system(f'isic-challenge-scoring classification {os.path.join(abs_path, "csvs", gtFile)} {os.path.join(abs_path, "csvs", f"TemporaryResults-{gtFile}")} > {os.path.join(abs_path, "txts", gtFile.split(".csv")[0]+"results.txt")}')
        # auc, balanced_accuracy = wandb_table(f'{os.path.join(abs_path, "txts", gtFile.split(".csv")[0]+"results.txt")}', epoch, num_classes)

        val_loss = sum(loss_acc) / float(test_loader.__len__())
        detection_accuracy = round(100*correct/total, 2)

        # if fold_index is None:
        #     wandb.log({'Val Set Loss': val_loss, 'epoch': epoch})
        #     wandb.log({'Balanced Accuracy': balanced_accuracy, 'epoch': epoch})
        #     wandb.log({'AUC': auc, 'epoch': epoch})
        # else:
        #     wandb.log({f'Val Set Loss - Fold {fold_index}': val_loss, 'epoch': epoch})
        #     wandb.log({f'Balanced Accuracy - Fold {fold_index}': balanced_accuracy, 'epoch': epoch})
        #     wandb.log({f'AUC Fold - {fold_index}': auc, 'epoch': epoch})

        wandb.log({'Val Set Loss': val_loss, 'epoch': epoch})

    # return val_loss, auc, balanced_accuracy
    return val_loss, detection_accuracy


def train(args):

    use_wandb = True
    device = torch.device(f'cuda')

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
        wandb.init(name=checkpointFileName, entity='ferles')

    if exclude_class is None:
        train_loader, val_loader, test_loader, columns = oversampling_loaders_custom(csvfiles=[traincsv, testcsv], train_batch_size=32, val_batch_size=16, gtFile=gtFileName)
    else:
        train_loader, val_loader, test_loader, columns = oversampling_loaders_exclude_class_custom_no_gts(csvfiles=[traincsv, testcsv], train_batch_size=32, val_batch_size=16, gtFile=gtFileName, exclude_class=exclude_class)
    model = build_model(args, rot=True)
    model = nn.DataParallel(model).to(device)

    epochs = 40
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1.25e-2, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    test_loss, best_val_detection_accuracy, test_detection_accuracy = 10, 0, 0

    train_loss, val_loss, balanced_accuracies = [], [], []

    # early_stopping = True
    early_stopping = False

    early_stopping_cnt = 0
    for epoch in tqdm(range(epochs)):

        model.train()
        loss_acc = []

        for data in tqdm(train_loader):
            inputs, labels = data
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            _labels = torch.argmax(labels, dim=1)
            ce_loss = criterion(outputs, _labels)

            # Rotation Loss
            rot_gt = torch.cat((torch.zeros(inputs.size(0)), torch.ones(inputs.size(0)),
                                2*torch.ones(inputs.size(0)), 3*torch.ones(inputs.size(0))), 0).long().to(device)

            rot_inputs = inputs.detach().cpu().numpy()

            rot_inputs = np.concatenate((rot_inputs, np.rot90(rot_inputs, 1, axes=(2, 3)),
                                 np.rot90(rot_inputs, 2, axes=(2, 3)), np.rot90(rot_inputs, 3, axes=(2, 3))), 0)

            rot_inputs = torch.FloatTensor(rot_inputs)
            rot_preds = model(rot_inputs, rot=True)

            rot_loss = 0.5 * criterion(rot_preds, rot_gt.to(device))

            loss = ce_loss + rot_loss

            loss_acc.append(loss.item())
            loss.backward()
            optimizer.step()

        wandb.log({'Train Set Loss': sum(loss_acc) / float(train_loader.__len__()), 'epoch': epoch})
        wandb.log({'epoch': epoch}, commit=False)
        train_loss.append(sum(loss_acc) / float(train_loader.__len__()))
        loss_acc.clear()

        with torch.no_grad():
            model.eval()
            correct, total = 0, 0
            for data in tqdm(val_loader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                softmax_outputs = torch.softmax(outputs, 1)
                max_idx = torch.argmax(softmax_outputs, axis=1)
                _labels = torch.argmax(labels, dim=1)
                correct += (max_idx == _labels).sum().item()
                total += max_idx.size()[0]
                loss = criterion(outputs, _labels)
                loss_acc.append(loss.item())

            val_detection_accuracy = round(100*correct/total, 2)
            wandb.log({'Validation Detection Accuracy': val_detection_accuracy, 'epoch': epoch})

            if val_detection_accuracy > best_val_detection_accuracy:
                best_val_detection_accuracy = val_detection_accuracy
                test_loss, test_detection_accuracy = _test_set_eval(model, epoch, device, test_loader, out_classes, columns, gtFileName)

                if exclude_class is None:
                    checkpointFile = os.path.join(f'/raid/ferles/checkpoints/isic_classifiers/rot_isic-best-model.pth')
                else:
                    checkpointFile = os.path.join(f'/raid/ferles/checkpoints/isic_classifiers/rot_isic-_{exclude_class}-best-model.pth')
                if os.path.exists('/raid/ferles/'):
                    torch.save(model.state_dict(), checkpointFile)
                else:
                    torch.save(model.state_dict(), checkpointFile.replace('raid', 'home'))
            else:
                if early_stopping:
                    early_stopping_cnt +=1
                    if early_stopping_cnt == 3:
                        wandb.log({'Test Set Loss': test_loss, 'epoch': epoch})
                        wandb.log({'Detection Accuracy': test_detection_accuracy, 'epoch': epoch})
                        break

            if exclude_class is None and epoch == 20:
                break
            elif exclude_class == 'AK' and epoch == 19:
                break
            elif exclude_class == 'BCC' and epoch == 12:
                break
            elif exclude_class == 'BKL' and epoch == 15:
                break
            elif exclude_class == 'DF' and epoch == 10:
                break
            elif exclude_class == 'MEL' and epoch == 12:
                break
            elif exclude_class == 'NV' and epoch == 27:
                break
            elif exclude_class == 'SCC' and epoch == 10:
                break
            elif exclude_class == 'VASC' and epoch == 10:
                break

            wandb.log({'Test Set Loss': test_loss, 'epoch': epoch})
            wandb.log({'Detection Accuracy': test_detection_accuracy, 'epoch': epoch})
        # val_loss, auc, balanced_accuracy = _test_set_eval(model, epoch, device, val_loader, out_classes, columns, gtFileName)

        # if auc > best_auc:
        #     best_auc = auc
        #     checkpointFile = os.path.join(f'{abs_path}/checkpoints/rotation/{checkpointFileName}-best-auc-model.pth')
        #     torch.save(model.state_dict(), checkpointFile)

        # if balanced_accuracy > best_balanced_accuracy:
        #     best_balanced_accuracy = balanced_accuracy
        #     checkpointFile = os.path.join(f'{abs_path}/checkpoints/rotation/{checkpointFileName}-best-balanced-accuracy-model.pth')
        #     torch.save(model.state_dict(), checkpointFile)

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     checkpointFile = os.path.join(f'{abs_path}/checkpoints/rotation/{checkpointFileName}-best-val-loss-model.pth')
        #     torch.save(model.state_dict(), checkpointFile)
        #     early_stopping_cnt = 0

        scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DL Dermatology models')
    parser.add_argument('--config', help='Training Configurations', required=True)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)
    args = parser.parse_args()

    visible_divices = f"{args.device}, {args.device+1}"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_divices

    train(args)
