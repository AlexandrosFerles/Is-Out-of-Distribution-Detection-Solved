import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
from utils import build_model_with_checkpoint
from dataLoaders import get_temp_data_loaders
from tqdm import tqdm
import lib_generation
import argparse
import os
import random
import pickle
import ipdb

abs_path = '/home/ferles/Dermatology/medusa/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _find_threshold(train_scores, val_scores):

    train_scores = np.sort(train_scores)
    val_scores = np.sort(val_scores)

    acc, threshold = 0, 0
    index = val_scores.shape[0] - 1

    while(True):

        temp_threshold = val_scores[index]
        ind_ = np.zeros(train_scores.shape)
        ood_ = np.zeros(val_scores.shape)
        ind_[np.argwhere(train_scores > temp_threshold)] = 1
        ood_[np.argwhere(val_scores < temp_threshold)] = 1

        temp_acc = np.sum(ind_) / ind_.shape[0] + (np.sum(ood_) / ood_.shape[0])
        if temp_acc > acc:
            acc, threshold = temp_acc, temp_threshold

        if temp_threshold < train_scores[0]:
            break

    return acc, threshold


def _score_classification_accuracy(model, testloader, dataset, genOdin=False):

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():

        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            if genOdin:
                outputs, h, g = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = round(100*correct/total, 2)
    print(f'Accuracy on {dataset}: {accuracy}%')


def _score_npzs(ind, ood, threshold):

    y_known, y_novel = np.ones(ind.shape[0]), np.zeros(ood.shape[0])
    X, y = np.append(ind, ood), np.append(y_known, y_novel)

    fpr, tpr, _ = roc_curve(y, X)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(100*roc_auc, 2)

    perc = 0.95
    sorted_ = np.sort(ind)
    fpr_threshold = sorted_[int((1-perc)*ind.shape[0])]

    ind_ = np.zeros(ind.shape)
    ood_ = np.zeros(ood.shape)
    ind_[np.argwhere(ind > fpr_threshold)] = 1
    ood_[np.argwhere(ood > fpr_threshold)] = 1

    X = np.append(ind_, ood_)
    bool_X = np.atleast_1d(X.astype(np.bool))
    bool_y = np.atleast_1d(y.astype(np.bool))

    tn = np.count_nonzero(~bool_X & ~bool_y)
    fp = np.count_nonzero(bool_X & ~bool_y)

    fpr = round(100*fp/(fp+tn), 2)

    ind_ = np.zeros(ind.shape)
    ood_ = np.zeros(ood.shape)
    ind_[np.argwhere(ind > threshold)] = 1
    ood_[np.argwhere(ood < threshold)] = 1

    acc = np.sum(ind_) / ind_.shape[0] + (np.sum(ood_) / ood_.shape[0])

    return roc_auc, fpr, acc


def _score_mahalanobis(ind, ood):

    y_known = np.ones(ind.shape[0])
    y_novel = np.zeros(ood.shape[0])

    X = np.append(ind, ood, axis=0)
    y = np.append(y_known, y_novel)

    lr = LogisticRegressionCV(n_jobs=-1, cv=5, max_iter=1000).fit(X, y)

    known_preds = lr.predict_proba(ind)[:, 1]
    novel_preds = lr.predict_proba(ood)[:, 1]

    fpr, tpr, _ = roc_curve(y, np.append(known_preds, novel_preds, axis=0))
    roc_auc = auc(fpr, tpr)

    perc = 0.95
    sorted_ = np.sort(known_preds)
    threshold = sorted_[int((1-perc)*known_preds.shape[0])]

    ind_ = np.zeros(known_preds.shape)
    ood_ = np.zeros(novel_preds.shape)
    ind_[np.argwhere(known_preds > threshold)] = 1
    ood_[np.argwhere(novel_preds > threshold)] = 1

    X = np.append(ind_, ood_)
    bool_X = np.atleast_1d(X.astype(np.bool))
    bool_y = np.atleast_1d(y.astype(np.bool))

    tn = np.count_nonzero(~bool_X & ~bool_y)
    fp = np.count_nonzero(bool_X & ~bool_y)

    fpr = round(fp/(fp+tn), 5)
    return lr, roc_auc, fpr


def _predict_mahalanobis(regressor, ind, ood):

    known_preds = regressor.predict_proba(ind)[:, 1]
    novel_preds = regressor.predict_proba(ood)[:, 1]
    y = np.append(np.ones(ind.shape[0]), np.zeros(ood.shape[0]), axis=0)

    fpr, tpr, _ = roc_curve(y, np.append(known_preds, novel_preds, axis=0))
    roc_auc = auc(fpr, tpr)

    perc = 0.95
    sorted_ = np.sort(known_preds)
    threshold = sorted_[int((1-perc)*known_preds.shape[0])]

    ind_ = np.zeros(known_preds.shape)
    ood_ = np.zeros(novel_preds.shape)
    ind_[np.argwhere(known_preds > threshold)] = 1
    ood_[np.argwhere(novel_preds > threshold)] = 1

    X = np.append(ind_, ood_)
    bool_X = np.atleast_1d(X.astype(np.bool))
    bool_y = np.atleast_1d(y.astype(np.bool))

    tn = np.count_nonzero(~bool_X & ~bool_y)
    fp = np.count_nonzero(bool_X & ~bool_y)

    fpr = round(fp/(fp+tn), 5)
    return roc_auc, fpr


def _baseline(model, loaders, device, ind_dataset, ood_dataset, monte_carlo_steps=1, exclude_class=None, score_ind=True):

    model.eval()

    if monte_carlo_steps > 1:
        model._dropout.train()
    val_loader, ood_loader = loaders

    if score_ind:
        ind_len = 0
        ind = np.zeros(val_loader.batch_size*val_loader.__len__())
        for index, data in enumerate(val_loader):

            images, _ = data
            images = images.to(device)

            outputs = model(images)
            softmax_outputs = torch.softmax(outputs, 1)
            top_class_probability = torch.max(softmax_outputs, axis=1)[0]

            ind[index*val_loader.batch_size:index*val_loader.batch_size + top_class_probability.size()[0]] = top_class_probability.detach().cpu().numpy()
            ind_len += top_class_probability.size()[0]

        if monte_carlo_steps > 1:
            for _ in range(monte_carlo_steps-1):
                for index, data in enumerate(val_loader):

                    images, _ = data
                    images = images.to(device)

                    outputs = model(images)
                    softmax_outputs = torch.softmax(outputs, 1)
                    top_class_probability = torch.max(softmax_outputs, axis=1)[0]

                    ind[index*val_loader.batch_size:index*val_loader.batch_size + top_class_probability.size()[0]] += top_class_probability.detach().cpu().numpy()

        ind = ind[:ind_len]

    ood_len = 0
    ood = np.zeros(ood_loader.batch_size*ood_loader.__len__())
    for index, data in enumerate(ood_loader):

        images, _ = data
        images = images.to(device)

        outputs = model(images)
        softmax_outputs = torch.softmax(outputs, 1)
        top_class_probability = torch.max(softmax_outputs, axis=1)[0]

        ood[index*ood_loader.batch_size:index*ood_loader.batch_size + top_class_probability.size()[0]] = top_class_probability.detach().cpu().numpy()
        ood_len += top_class_probability.size()[0]

    if monte_carlo_steps > 1:
        for _ in range(monte_carlo_steps-1):
            for index, data in enumerate(ood_loader):
                images, _ = data
                images = images.to(device)

                outputs = model(images)
                softmax_outputs = torch.softmax(outputs, 1)
                top_class_probability = torch.max(softmax_outputs, axis=1)[0]

                ood[index*ood_loader.batch_size:index*ood_loader.batch_size + top_class_probability.size()[0]] += top_class_probability.detach().cpu().numpy()

    ood = ood[:ood_len]

    if monte_carlo_steps > 1:
        if score_ind:
            ind = ind / monte_carlo_steps
            ood = ood / monte_carlo_steps
        else:
            ood = ood / monte_carlo_steps

    if exclude_class is None:
        if monte_carlo_steps == 1:
            ind_savefile_name = f'npzs/baseline_{ind_dataset}.npz'
            ood_savefile_name = f'npzs/baseline_{ood_dataset}.npz'
        else:
            ind_savefile_name = f'npzs/baseline_{ind_dataset}_monte_carlo_{monte_carlo_steps}.npz'
            ood_savefile_name = f'npzs/baseline_{ood_dataset}_monte_carlo_{monte_carlo_steps}.npz'
    else:
        if monte_carlo_steps == 1:
            ind_savefile_name = f'npzs/baseline_{ind_dataset}_{exclude_class}.npz'
            ood_savefile_name = f'npzs/baseline_{ood_dataset}_{exclude_class}.npz'
        else:
            ind_savefile_name = f'npzs/baseline_{ind_dataset}_monte_carlo_{monte_carlo_steps}_{exclude_class}.npz'
            ood_savefile_name = f'npzs/baseline_{ood_dataset}_monte_carlo_{monte_carlo_steps}_{exclude_class}.npz'

    if score_ind:
        np.savez(ind_savefile_name, ind)
        np.savez(ood_savefile_name, ood)
        auc, fpr = _score_npzs(ind, ood)

        print('###############################################')
        print()
        print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name} and out-distribution ood scores to: {ood_savefile_name}')
        print()
        print('###############################################')
        print()
        if monte_carlo_steps == 1:
            print(f"Baseline results on {ind_dataset} (In) vs {ood_dataset} (Out):")
        else:
            print(f"Baseline results with MC dropout ({monte_carlo_steps} steps) on {ind_dataset} (In) vs {ood_dataset} (Out):")
        print()
        print(f'Area Under Receiver Operating Characteristic curve: {auc}')
        print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')

    else:
        np.savez(ood_savefile_name, ood)
        print('###############################################')
        print()
        print(f'Succesfully stored in-distribution out-distribution ood scores to: {ood_savefile_name}')
        print()
        print('###############################################')


def _create_fgsm_loader(val_loader):

    sample, gts = next(iter(val_loader))
    sizes = sample.size()
    len_ = 0
    ood_data_x = torch.zeros(size=(val_loader.__len__()*val_loader.batch_size, sizes[1], sizes[2], sizes[3]))
    ood_data_y = torch.zeros(val_loader.__len__()*val_loader.batch_size)
    fgsm_step = 0.1
    criterion = nn.CrossEntropyLoss()
    for index, data in enumerate(val_loader):

        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        input_var = torch.autograd.Variable(images, requires_grad=True)
        input_var = input_var.to(device)
        output = model(input_var)
        if len(labels.size()) > 1:
            labels = torch.argmax(labels, dim=1)
        loss = criterion(output, labels)
        loss.backward()

        sign_data_grad = input_var.grad.data.sign()
        perturbed_image = input_var + fgsm_step*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        ood_data_x[index*val_loader.batch_size:index*val_loader.batch_size+images.size(0)] = perturbed_image
        ood_data_y[index*val_loader.batch_size:index*val_loader.batch_size+images.size(0)] = labels
        len_ += images.size(0)

    ood_data_x, ood_data_y = ood_data_x[:len_], ood_data_y[:len_]
    fgsm_dataset = TensorDataset(ood_data_x, ood_data_y)
    fgsm_loader = DataLoader(fgsm_dataset, batch_size=val_loader.batch_size)

    return fgsm_loader


def _process(model, images, T, epsilon, device, criterion=nn.CrossEntropyLoss()):

    model.eval()
    if T == 1 and epsilon == 0:
        outputs = model(images.to(device))
        nnOutputs = outputs.detach().cpu().numpy()
    else:
        inputs = Variable(images.to(device), requires_grad=True)
        outputs = model(inputs)
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1).reshape(nnOutputs.shape[0], 1)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs), axis=1).reshape(nnOutputs.shape[0], 1)
        outputs = outputs / T

        maxIndexTemp = np.argmax(nnOutputs, axis=1)
        labels = Variable(torch.LongTensor([maxIndexTemp]).to(device))
        loss = criterion(outputs, labels[0])
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(inputs.data,  -epsilon, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / T

        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1).reshape(nnOutputs.shape[0], 1)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs), axis=1).reshape(nnOutputs.shape[0], 1)

    return nnOutputs


def _get_odin_scores(model, loader, T, epsilon, device, score_entropy=False):

    model.eval()

    len_ = 0
    arr = np.zeros(loader.batch_size*loader.__len__())
    for index, data in enumerate(loader):
        images, _ = data
        nnOutputs = _process(model, images, T, epsilon, device=device)
        top_class_probability = np.max(nnOutputs, axis=1)
        if not score_entropy:
            arr[index*loader.batch_size:index*loader.batch_size + top_class_probability.shape[0]] = top_class_probability
        else:
            entropy = -np.sum(np.log(nnOutputs) * nnOutputs, axis=1)
            arr[index*loader.batch_size:index*loader.batch_size + top_class_probability.shape[0]] = top_class_probability - entropy
        len_ += top_class_probability.shape[0]

    return arr[:len_]


def _odin(model, loaders, device, ind_dataset, ood_dataset, T=None, epsilon=None, with_fgsm=False, exclude_class=None, score_ind=True):

    model.eval()

    if with_fgsm:
        val_loader, test_loader, ood_loader = loaders
        fgsm_loader = _create_fgsm_loader(val_loader)
    else:
        test_loader, ood_loader = loaders

    if T is not None:
        if epsilon is not None:
            if score_ind:
                ind = _get_odin_scores(model, test_loader, T, epsilon, device=device)
                ood = _get_odin_scores(model, ood_loader, T, epsilon, device=device)

                if exclude_class is None:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{T}_epsilon{epsilon}.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{T}_epsilon{epsilon}.npz'
                else:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{T}_epsilon{epsilon}_{exclude_class}.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{T}_epsilon{epsilon}_{exclude_class}.npz'
                np.savez(ind_savefile_name, ind)
                np.savez(ood_savefile_name, ood)
                auc, fpr = _score_npzs(ind, ood)
                print('###############################################')
                print()
                print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name} and out-distribution ood scores to: {ood_savefile_name}')
                print()
                print('###############################################')
                print()
                print(f"Odin results on {ind_dataset} (In) vs {ood_dataset} (Out) with T={T} and epsilon={epsilon}:")
                print()
                print(f'Area Under Receiver Operating Characteristic curve: {auc}')
                print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')
            else:
                ood = _get_odin_scores(model, ood_loader, T, epsilon, device=device)
                if exclude_class is None:
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{T}_epsilon{epsilon}.npz'
                else:
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{T}_epsilon{epsilon}_{exclude_class}.npz'
                np.savez(ood_savefile_name, ood)
                print('###############################################')
                print()
                print(f'Succesfully stored in-distribution ood scores to out-distribution ood scores to: {ood_savefile_name}')
                print()
                print('###############################################')
                print()
        else:
            if with_fgsm:
                best_fpr = 100
                best_epsilon = 0
                for epsilon in tqdm(np.arange(0, 0.004, 0.004/21, float).tolist()):
                    ind = _get_odin_scores(model, val_loader, T, epsilon, device=device)
                    ood = _get_odin_scores(model, fgsm_loader, T, epsilon, device=device)

                    _, fpr = _score_npzs(ind, ood)
                    if fpr <= best_fpr:
                        best_fpr = fpr
                        best_epsilon = epsilon

                print('###############################################')
                print()
                print(f'Selected temperature: {T}, selected epsilon: {best_epsilon}')
                print()
                ind = _get_odin_scores(model, test_loader, device, T, best_epsilon, device=device)
                ood = _get_odin_scores(model, ood_loader, device, T, best_epsilon, device=device)
                if exclude_class is None:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{T}_epsilon{best_epsilon}_fgsm.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{T}_epsilon{best_epsilon}_fgsm.npz'
                else:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{T}_epsilon{best_epsilon}_fgsm_{exclude_class}.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{T}_epsilon{best_epsilon}_fgsm_{exclude_class}.npz'
                np.savez(ind_savefile_name, ind)
                np.savez(ood_savefile_name, ood)
                auc, fpr = _score_npzs(ind, ood)
                print('###############################################')
                print()
                print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name} and out-distribution ood scores to: {ood_savefile_name}')
                print()
                print('###############################################')
                print()
                print(f"Odin results on {ind_dataset} (In) vs {ood_dataset} (Out) with T={T} and epsilon={best_epsilon}:")
                print()
                print(f'Area Under Receiver Operating Characteristic curve: {auc}')
                print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')

            else:
                best_epsilon, best_fpr = 0, 100
                for epsilon in tqdm(np.arange(0, 0.004, 0.004/21, float).tolist()):
                    ind = _get_odin_scores(model, test_loader, T, epsilon, device=device)
                    ood = _get_odin_scores(model, ood_loader, T, epsilon, device=device)
                    _, fpr = _score_npzs(ind, ood)
                    if fpr <= best_fpr:
                        best_fpr = fpr
                        best_epsilon = epsilon

                print('###############################################')
                print()
                print(f'Selected temperature: {T}, selected epsilon: {best_epsilon}')
                print()

                ind = _get_odin_scores(model, test_loader, T, best_epsilon, device=device)
                ood = _get_odin_scores(model, ood_loader, T, best_epsilon, device=device)
                if exclude_class is None:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{T}_epsilon{best_epsilon}.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{T}_epsilon{best_epsilon}.npz'
                else:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{T}_epsilon{best_epsilon}_{exclude_class}.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{T}_epsilon{best_epsilon}_{exclude_class}.npz'
                np.savez(ind_savefile_name, ind)
                np.savez(ood_savefile_name, ood)
                auc, fpr = _score_npzs(ind, ood)
                print('###############################################')
                print()
                print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name} and out-distribution ood scores to: {ood_savefile_name}')
                print()
                print('###############################################')
                print()
                print(f"Odin results on {ind_dataset} (In) vs {ood_dataset} (Out) with T={T} and epsilon={best_epsilon}:")
                print(f'Area Under Receiver Operating Characteristic curve: {auc}')
                print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')
    else:
        if epsilon is not None:
            if with_fgsm:
                best_fpr = 100
                best_T = 1

                for T in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
                    ind = _get_odin_scores(model, val_loader, T, epsilon, device=device)
                    ood = _get_odin_scores(model, fgsm_loader, T, epsilon, device=device)

                    _, fpr = _score_npzs(ind, ood)
                    if fpr < best_fpr:
                        best_fpr = fpr
                        best_T = T

                print('###############################################')
                print()
                print(f'Selected temperature: {best_T}, selected epsilon: {epsilon}')
                print()

                ind = _get_odin_scores(model, test_loader, device, best_T, epsilon)
                ood = _get_odin_scores(model, ood_loader, device, best_T, epsilon)
                if exclude_class is None:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{best_T}_epsilon{epsilon}_fgsm.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{best_T}_epsilon{epsilon}_fgsm.npz'
                else:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{best_T}_epsilon{epsilon}_fgsm_{exclude_class}.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{best_T}_epsilon{epsilon}_fgsm_{exclude_class}.npz'
                np.savez(ind_savefile_name, ind)
                np.savez(ood_savefile_name, ood)
                auc, fpr = _score_npzs(ind, ood)
                print('###############################################')
                print()
                print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name} and out-distribution ood scores to: {ood_savefile_name}')
                print()
                print('###############################################')
                print()
                print(f"Odin results on {ind_dataset} (In) vs {ood_dataset} (Out) with T={best_T} and epsilon={epsilon}:")
                print(f'Area Under Receiver Operating Characteristic curve: {auc}')
                print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')
            else:
                best_T, best_fpr = 0, 100
                for T in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
                    ind = _get_odin_scores(model, test_loader, T, epsilon, device=device)
                    ood = _get_odin_scores(model, ood_loader, T, epsilon, device=device)
                    _, fpr = _score_npzs(ind, ood)
                    if fpr <= best_fpr:
                        best_fpr = fpr
                        best_T = T

                print('###############################################')
                print()
                print(f'Selected temperature: {best_T}, selected epsilon: {epsilon}')
                print()

                ind = _get_odin_scores(model, test_loader, best_T, epsilon, device=device)
                ood = _get_odin_scores(model, ood_loader, best_T, epsilon, device=device)
                if exclude_class is None:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{best_T}_epsilon{epsilon}.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{best_T}_epsilon{epsilon}.npz'
                else:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{best_T}_epsilon{epsilon}_{exclude_class}.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{best_T}_epsilon{epsilon}_{exclude_class}.npz'
                np.savez(ind_savefile_name, ind)
                np.savez(ood_savefile_name, ood)
                auc, fpr = _score_npzs(ind, ood)
                print('###############################################')
                print()
                print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name} and out-distribution ood scores to: {ood_savefile_name}')
                print()
                print('###############################################')
                print()
                print(f"Odin results on {ind_dataset} (In) vs {ood_dataset} (Out) with T={best_T} and epsilon={epsilon}:")
                print(f'Area Under Receiver Operating Characteristic curve: {auc}')
                print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')
        else:
            if with_fgsm:
                best_fpr = 100
                best_epsilon, best_T = 0, 1
                for T in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
                    for epsilon in tqdm(np.arange(0, 0.004, 0.004/21, float).tolist()):

                        ind = _get_odin_scores(model, val_loader, T, epsilon, device=device)
                        ood = _get_odin_scores(model, fgsm_loader, T, epsilon, device=device)

                        _, fpr = _score_npzs(ind, ood)
                        if fpr < best_fpr:
                            best_fpr = fpr
                            best_epsilon = epsilon
                            best_T = T

                print('###############################################')
                print()
                print(f'Selected temperature: {best_T}, selected epsilon: {best_epsilon}')
                print()

                ind = _get_odin_scores(model, test_loader, best_T, best_epsilon, device=device)
                ood = _get_odin_scores(model, ood_loader, best_T, best_epsilon, device=device)
                if exclude_class is None:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{best_T}_epsilon{best_epsilon}.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{best_T}_epsilon{best_epsilon}.npz'
                else:
                    ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{best_T}_epsilon{best_epsilon}_{exclude_class}.npz'
                    ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{best_T}_epsilon{best_epsilon}_{exclude_class}.npz'
                np.savez(ind_savefile_name, ind)
                np.savez(ood_savefile_name, ood)
                auc, fpr = _score_npzs(ind, ood)
                print('###############################################')
                print()
                print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name} and out-distribution ood scores to: {ood_savefile_name}')
                print()
                print('###############################################')
                print()
                print(f"Odin results on {ind_dataset} (In) vs {ood_dataset} (Out) with T={best_T} and epsilon={best_epsilon}:")
                print(f'Area Under Receiver Operating Characteristic curve: {auc}')
                print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')
            else:
                for T in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
                    for epsilon in tqdm(np.arange(0, 0.004, 0.004/21, float).tolist()):

                        if T==1 and epsilon==0:
                            # baseline, skipping
                            continue
                        ind = _get_odin_scores(model, test_loader, device, T, epsilon, device=device)
                        ood = _get_odin_scores(model, ood_loader, device, T, epsilon, device=device)

                        if exclude_class is None:
                            ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{T}_epsilon{epsilon}.npz'
                            ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{T}_epsilon{epsilon}.npz'
                        else:
                            ind_savefile_name = f'npzs/odin_{ind_dataset}_temperature_{T}_epsilon{epsilon}_{exclude_class}.npz'
                            ood_savefile_name = f'npzs/odin_{ood_dataset}_temperature_{T}_epsilon{epsilon}_{exclude_class}.npz'
                        np.savez(ind_savefile_name, ind)
                        np.savez(ood_savefile_name, ood)
                        auc, fpr = _score_npzs(ind, ood)
                        print('###############################################')
                        print()
                        print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name} and out-distribution ood scores to {ood_savefile_name}')
                        print()
                        print('###############################################')
                        print()
                        print(f"Odin results on {ind_dataset} (In) vs {ood_dataset} (Out) with T={T} and epsilon={epsilon}:")
                        print(f'Area Under Receiver Operating Characteristic curve: {auc}')
                        print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')


def _generate_Mahalanobis(model, loaders, device, ind_dataset, ood_dataset, num_classes=10, exclude_class=None, model_type='eb0', score_ind=True):

    model.eval()

    train_loader, val_loader, test_loader, out_test_loader = loaders
    fgsm_loader = _create_fgsm_loader(val_loader)

    temp_x = torch.rand(2, 3, 224, 224).to(device)
    temp_x = Variable(temp_x)
    temp_x = temp_x.to(device)
    if model_type == 'eb0':
        idxs = [0, 2, 4, 7, 10, 14, 15]
        x, features = model.extract_features(temp_x, mode='all')
    else:
        # TODO: In case you wish to evaluate other models, you need to define a proper way to get middle level features
        pass
    features = [features[idx] for idx in idxs] + [x]
    num_output = len(features)
    feature_list = np.empty(num_output)
    count = 0
    for out in features:
        feature_list[count] = out.size(1)
        count += 1

    sample_mean, precision = lib_generation.sample_estimator(model, num_classes, feature_list, train_loader, device=device)

    best_fpr = 100
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    if score_ind:
        best_magnitudes, best_fprs, regressors = [], [], []
        for magnitude in m_list:
            for i in range(num_output):
                M_val = lib_generation.get_Mahalanobis_score(model, val_loader, num_classes, sample_mean, precision, i, magnitude, device=device)
                M_val = np.asarray(M_val, dtype=np.float32)
                if i == 0:
                    Mahalanobis_val = M_val.reshape((M_val.shape[0], -1))
                else:
                    Mahalanobis_val = np.concatenate((Mahalanobis_val, M_val.reshape((M_val.shape[0], -1))), axis=1)

            for i in range(num_output):
                M_fgsm = lib_generation.get_Mahalanobis_score(model, fgsm_loader, num_classes, sample_mean, precision, i, magnitude, device=device)
                M_fgsm = np.asarray(M_fgsm, dtype=np.float32)
                if i == 0:
                    Mahalanobis_fgsm = M_fgsm.reshape((M_fgsm.shape[0], -1))
                else:
                    Mahalanobis_fgsm = np.concatenate((Mahalanobis_fgsm, M_fgsm.reshape((M_fgsm.shape[0], -1))), axis=1)

            Mahalanobis_val = np.asarray(Mahalanobis_val, dtype=np.float32)
            Mahalanobis_fgsm = np.asarray(Mahalanobis_fgsm, dtype=np.float32)

            regressor, _, fpr = _score_mahalanobis(Mahalanobis_val, Mahalanobis_fgsm)
            with open(f'lr_pickles/logistic_regressor_{exclude_class}_{magnitude}.pickle', 'wb') as lrp:
                pickle.dump(regressor, lrp, protocol=pickle.HIGHEST_PROTOCOL)

            if fpr < best_fpr:
                best_fpr = fpr
                best_magnitudes = [magnitude]
                regressors = [regressor]
            elif fpr == best_fpr:
                best_magnitudes.append(magnitude)
                regressors.append(regressor)

        print('###############################################')
        print()
        print(f'Selected magnitudes: {best_magnitudes}')
        print()

        for (best_magnitude, regressor) in zip(best_magnitudes, regressors):
            for i in range(num_output):
                M_test = lib_generation.get_Mahalanobis_score(model, test_loader, num_classes, sample_mean, precision, i, best_magnitude, device=device)
                M_test = np.asarray(M_test, dtype=np.float32)
                if i == 0:
                    Mahalanobis_test = M_test.reshape((M_test.shape[0], -1))
                else:
                    Mahalanobis_test = np.concatenate((Mahalanobis_test, M_test.reshape((M_test.shape[0], -1))), axis=1)

            for i in range(num_output):
                M_ood = lib_generation.get_Mahalanobis_score(model, out_test_loader, num_classes, sample_mean, precision, i, best_magnitude, device=device)
                M_ood = np.asarray(M_ood, dtype=np.float32)
                if i == 0:
                    Mahalanobis_ood = M_ood.reshape((M_ood.shape[0], -1))
                else:
                    Mahalanobis_ood = np.concatenate((Mahalanobis_ood, M_ood.reshape((M_ood.shape[0], -1))), axis=1)

            Mahalanobis_test = np.asarray(Mahalanobis_test, dtype=np.float32)
            Mahalanobis_ood = np.asarray(Mahalanobis_ood, dtype=np.float32)

            if exclude_class is None:
                ind_savefile_name = f'npzs/Mahalanobis_{ind_dataset}_{best_magnitude}.npz'
                ood_savefile_name = f'npzs/Mahalanobis_{ood_dataset}_{best_magnitude}.npz'
            else:
                ind_savefile_name = f'npzs/Mahalanobis_{ind_dataset}_{best_magnitude}_{exclude_class}.npz'
                ood_savefile_name = f'npzs/Mahalanobis_{ood_dataset}_{best_magnitude}_{exclude_class}.npz'
            np.savez(ind_savefile_name, Mahalanobis_test)
            np.savez(ood_savefile_name, Mahalanobis_ood)
            auc, fpr = _predict_mahalanobis(regressor, Mahalanobis_test, Mahalanobis_ood)
            print('###############################################')
            print()
            print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name} and out-distribution ood scores to {ood_savefile_name}')
            print()
            print('###############################################')
            print()
            print(f'Mahalanobis results on {ind_dataset} (In) vs {ood_dataset}:')
            print(f'Area Under Receiver Operating Characteristic curve: {auc}')
            print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')
            print('###############################################')
            print()
    else:
        for magnitude in m_list:
            for i in range(num_output):
                M_ood = lib_generation.get_Mahalanobis_score(model, out_test_loader, num_classes, sample_mean, precision, i, magnitude, device=device)
                M_ood = np.asarray(M_ood, dtype=np.float32)
                if i == 0:
                    Mahalanobis_ood = M_ood.reshape((M_ood.shape[0], -1))
                else:
                    Mahalanobis_ood = np.concatenate((Mahalanobis_ood, M_ood.reshape((M_ood.shape[0], -1))), axis=1)

            Mahalanobis_ood = np.asarray(Mahalanobis_ood, dtype=np.float32)

            if exclude_class is None:
                ood_savefile_name = f'npzs/Mahalanobis_{ood_dataset}_{magnitude}.npz'
            else:
                ood_savefile_name = f'npzs/Mahalanobis_{ood_dataset}_{magnitude}_{exclude_class}.npz'
            np.savez(ood_savefile_name, Mahalanobis_ood)
            print('###############################################')
            print()
            print(f'Succesfully stored in-distribution and out-distribution ood scores to {ood_savefile_name}')
            print()
            print('###############################################')


def _predict_rotations(model, loader, num_classes, device):

    model.eval()
    uniform = torch.zeros(num_classes)
    for i in range(uniform.size()[0]):
        uniform[i] = 1.0 / uniform.size()[0]

    uniform = uniform.to(device)

    numpy_array_full = np.zeros(loader.batch_size*loader.__len__())
    numpy_array_kl_div = np.zeros(loader.batch_size*loader.__len__())
    numpy_array_rot_score = np.zeros(loader.batch_size*loader.__len__())
    for index, data in tqdm(enumerate(loader)):

        images, _ = data
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            log_softmax_outputs = torch.log_softmax(outputs, 1)
            kl_div = F.kl_div(log_softmax_outputs, uniform)

            # Rotation Loss
            rot_gt = torch.cat((torch.zeros(images.size(0)),
                                torch.ones(images.size(0)),
                                2*torch.ones(images.size(0)),
                                3*torch.ones(images.size(0))), 0).long().to(device)

            rot_images = images.detach().cpu().numpy()
            rot_images = np.concatenate((rot_images, np.rot90(rot_images, 1, axes=(2, 3)),
                                         np.rot90(rot_images, 2, axes=(2, 3)), np.rot90(rot_images, 3, axes=(2, 3))), 0)

            rot_images = torch.FloatTensor(rot_images)
            rot_images = rot_images.to(device)

            rot_preds = model(rot_images, rot=True)
            ce_rot = F.cross_entropy(rot_preds, rot_gt)

            numpy_array_kl_div[index] = kl_div.item()
            numpy_array_rot_score[index] = - 0.25 * ce_rot.item()
            anomaly_score = kl_div.item() - 0.25 * ce_rot.item()
            numpy_array_full[index] = anomaly_score

    return numpy_array_kl_div, numpy_array_rot_score, numpy_array_full


def _rotation(model, loaders, device, ind_dataset, ood_dataset, num_classes, exclude_class=None):

    val_loader, ood_loader = loaders
    ind_kl_div, ind_rot_score, ind_full = _predict_rotations(model, val_loader, num_classes, device=device)
    ood_kl_div, ood_rot_score, ood_full = _predict_rotations(model, ood_loader, num_classes, device=device)

    if exclude_class is None:
        ind_savefile_name_kl_div = f'npzs/self_supervision_{ind_dataset}_kl_div.npz'
        ind_savefile_name_rot_score = f'npzs/self_supervision_{ind_dataset}_rot_score.npz'
        ind_savefile_name_full = f'npzs/self_supervision_{ind_dataset}_full.npz'
        ood_savefile_name_kl_div = f'npzs/self_supervision_{ood_dataset}_kl_div.npz'
        ood_savefile_name_rot_score = f'npzs/self_supervision_{ood_dataset}_rot_score.npz'
        ood_savefile_name_full = f'npzs/self_supervision_{ood_dataset}_full.npz'
    else:
        ind_savefile_name_kl_div = f'npzs/self_supervision_{ind_dataset}_kl_div_{exclude_class}.npz'
        ind_savefile_name_rot_score = f'npzs/self_supervision_{ind_dataset}_rot_score_{exclude_class}.npz'
        ind_savefile_name_full = f'npzs/self_supervision_{ind_dataset}_full_{exclude_class}.npz'
        ood_savefile_name_kl_div = f'npzs/self_supervision_{ood_dataset}_kl_div_{exclude_class}.npz'
        ood_savefile_name_rot_score = f'npzs/self_supervision_{ood_dataset}_rot_score_{exclude_class}.npz'
        ood_savefile_name_full = f'npzs/self_supervision_{ood_dataset}_full_{exclude_class}.npz'
    np.savez(ind_savefile_name_kl_div, ind_kl_div)
    np.savez(ind_savefile_name_rot_score, ind_rot_score)
    np.savez(ind_savefile_name_full, ind_full)
    np.savez(ood_savefile_name_kl_div, ood_kl_div)
    np.savez(ood_savefile_name_rot_score, ood_rot_score)
    np.savez(ood_savefile_name_full, ood_full)
    auc_kl_div, fpr_kl_div = _score_npzs(ind_kl_div, ood_kl_div)
    auc_rot_score, fpr_rot_score = _score_npzs(ind_rot_score, ood_rot_score)
    auc_full, fpr_full = _score_npzs(ind_full, ood_full)
    print('###############################################')
    print()
    print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name_kl_div} and out-distribution ood scores to {ood_savefile_name_kl_div}')
    print()
    print('###############################################')
    print()
    print(f"Self-Supervision results (KL-Divergence) on {ind_dataset} (In) vs {ood_dataset}:")
    print(f'Area Under Receiver Operating Characteristic curve: {auc_kl_div}')
    print(f'False Positive Rate @ 95% True Positive Rate: {fpr_kl_div}')
    print('###############################################')
    print()
    print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name_rot_score} and out-distribution ood scores to {ood_savefile_name_rot_score}')
    print()
    print('###############################################')
    print()
    print(f"Self-Supervision results (Rotation Score) on {ind_dataset} (In) vs {ood_dataset}:")
    print(f'Area Under Receiver Operating Characteristic curve: {auc_rot_score}')
    print(f'False Positive Rate @ 95% True Positive Rate: {fpr_rot_score}')
    print('###############################################')
    print()
    print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name_full} and out-distribution ood scores to {ood_savefile_name_full}')
    print()
    print('###############################################')
    print()
    print(f"Self-Supervision results on {ind_dataset} (In) vs {ood_dataset}:")
    print(f'Area Under Receiver Operating Characteristic curve: {auc_full}')
    print(f'False Positive Rate @ 95% True Positive Rate: {fpr_full}')


def _ensemble_inference(model_checkpoints, loaders, device, out_classes, ind_dataset, ood_dataset, T=1000, epsilon=0.002, mode='accuracy', scaling =True):

    test_loader, ood_loader = loaders
    index = 0
    for model_checkpoint in tqdm(model_checkpoints):
        model = build_model_with_checkpoint('eb0', model_checkpoint, device, out_classes=out_classes)
        model.eval()
        if scaling:
            if index == 0:
                ind = _get_odin_scores(model, test_loader, T, epsilon, device=device, score_entropy=True)
                ood = _get_odin_scores(model, ood_loader, T, epsilon, device=device, score_entropy=True)
            else:
                ind += _get_odin_scores(model, test_loader, T, epsilon, device=device, score_entropy=True)
                ood += _get_odin_scores(model, ood_loader, T, epsilon, device=device, score_entropy=True)
        else:
            if index == 0:
                ind = _get_odin_scores(model, test_loader, T=1, epsilon=0, device=device, score_entropy=True)
                ood = _get_odin_scores(model, ood_loader, T=1, epsilon=0, device=device, score_entropy=True)
            else:
                ind += _get_odin_scores(model, test_loader, T=1, epsilon=0, device=device, score_entropy=True)
                ood += _get_odin_scores(model, ood_loader, T=1, epsilon=0, device=device, score_entropy=True)
        index += 1

    ind_savefile_name = f'npzs/ensemble_{ind_dataset}_mode_{mode}.npz'
    ood_savefile_name = f'npzs/ensemble_{ood_dataset}_mode_{mode}.npz'

    ind = ind / (index-1)
    ood = ood / (index-1)
    np.savez(ind_savefile_name, ind)
    np.savez(ood_savefile_name, ood)
    auc, fpr = _score_npzs(ind, ood)
    print('###############################################')
    print()
    print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name} and out-distribution ood scores to {ood_savefile_name}')
    print()
    print('###############################################')
    print()
    print(f"Leave-Out Ensemble results ({mode}) on {ind_dataset} (In) vs {ood_dataset}:")
    print(f'Area Under Receiver Operating Characteristic curve: {auc}')
    print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')


def _process_gen_odin(model, images, epsilon, criterion=nn.CrossEntropyLoss()):

    model.eval()
    inputs = Variable(images.to(device), requires_grad=True)
    outputs, _, _ = model(inputs)
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1).reshape(nnOutputs.shape[0], 1)
    nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs), axis=1).reshape(nnOutputs.shape[0], 1)

    maxIndexTemp = np.argmax(nnOutputs, axis=1)
    labels = Variable(torch.LongTensor([maxIndexTemp]).to(device))
    loss = criterion(outputs, labels[0])
    loss.backward()

    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    tempInputs = torch.add(inputs.data,  -epsilon, gradient)
    o, h, g = model(Variable(tempInputs))

    return np.max(o.detach().cpu().numpy(), axis=1), np.max(h.detach().cpu().numpy(), axis=1), g.detach().cpu().numpy()


def _gen_odin_temp(model, loaders, ind_dataset, device):

    model.eval()
    test_loader, ood_loader = loaders
    _score_classification_accuracy(model, testloader, dataset=ind_dataset, genOdin=True)

    max_h_scores_ind = np.zeros(test_loader.batch_size*test_loader.__len__())
    len_ = 0
    for index, data in enumerate(test_loader):

        images, _ = data
        images = images.to(device)
        _, h, _ = model(images)

        max_h_scores_ind[index*test_loader.batch_size:index*test_loader.batch_size + h.shape[0]] = np.max(h.detach().cpu().numpy(), axis=1)
        len_ += h.shape[0]

    max_h_scores_ind = max_h_scores_ind[:len_]

    max_h_scores_ood = np.zeros(ood_loader.batch_size*ood_loader.__len__())
    len_ = 0
    for index, data in enumerate(ood_loader):

        images, _ = data
        images = images.to(device)
        _, h, _ = model(images)

        max_h_scores_ood[index*ood_loader.batch_size:index*ood_loader.batch_size + h.shape[0]] = np.max(h.detach().cpu().numpy(), axis=1)
        len_ += h.shape[0]

    max_h_scores_ood = max_h_scores_ood[:len_]
    auc, fpr = _score_npzs(max_h_scores_ind, max_h_scores_ood)

    print('#########################################')
    print(f'Epsilon: {0}')
    print('-----------------------------------------')
    print(f'AUC: {auc}')
    print(f'FPR: {fpr}')
    print('#########################################')

    epsilons = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
    for epsilon in epsilons:

        max_h_scores_ind = np.zeros(test_loader.batch_size*test_loader.__len__())
        len_ = 0
        for index, data in enumerate(test_loader):

            images, _ = data
            images = images.to(device)
            _, h, _ = _process_gen_odin(model, images, epsilon)

            max_h_scores_ind[index*test_loader.batch_size:index*test_loader.batch_size + h.shape[0]] = h
            len_ += h.shape[0]

        max_h_scores_ind = max_h_scores_ind[:len_]

        max_h_scores_ood = np.zeros(ood_loader.batch_size*ood_loader.__len__())
        len_ = 0
        for index, data in enumerate(ood_loader):

            images, _ = data
            images = images.to(device)
            _, h, _ = _process_gen_odin(model, images, epsilon)

            max_h_scores_ood[index*ood_loader.batch_size:index*ood_loader.batch_size + h.shape[0]] = h
            len_ += h.shape[0]

        max_h_scores_ood = max_h_scores_ood[:len_]

        auc, fpr = _score_npzs(max_h_scores_ind, max_h_scores_ood)

        print('#########################################')
        print(f'Epsilon: {epsilon}')
        print('-----------------------------------------')
        print(f'AUC: {auc}')
        print(f'FPR: {fpr}')
        print('#########################################')



def _gen_odin_inference(model, loaders, ind_dataset, ood_dataset, mode, exclude_class=None):

    model.eval()
    val_loader_in, test_loader_in, ood_loader = loaders
    epsilons = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]

    best_epsilon_g, best_epsilon_h, best_epsilon_o = 0, 0, 0
    best_score_g, best_score_h, best_score_o = 0, 0, 0

    for epsilon in epsilons:
        len_ = 0
        max_o = np.zeros(val_loader_in.batch_size*val_loader_in.__len__())
        max_h = np.zeros(val_loader_in.batch_size*val_loader_in.__len__())
        sigmoid = np.zeros(val_loader_in.batch_size*val_loader_in.__len__())
        for index, data in enumerate(val_loader_in):

            images, _ = data
            images = images.to(device)
            o, h, g = _process_gen_odin(model, images, epsilon)

            max_o[index*val_loader_in.batch_size:index*val_loader_in.batch_size + o.shape[0]] = o
            max_h[index*val_loader_in.batch_size:index*val_loader_in.batch_size + o.shape[0]] = h
            sigmoid[index*val_loader_in.batch_size:index*val_loader_in.batch_size + o.shape[0]] = np.squeeze(g)
            len_ += o.shape[0]

        o = o[:len_]
        h = h[:len_]
        g = g[:len_]

        score_g = np.average(g)
        if score_g > best_score_g:
            best_score_g = score_g
            best_epsilon_g = epsilon

        score_h = np.average(h)
        if score_h > best_score_h:
            best_score_h = score_h
            best_epsilon_h = epsilon

        score_o = np.average(o)
        if score_o > best_score_o:
            best_score_o = score_o
            best_epsilon_o = epsilon

    len_ = 0
    max_o_scores_ind = np.zeros(test_loader_in.batch_size*test_loader_in.__len__())
    max_h_scores_ind = np.zeros(test_loader_in.batch_size*test_loader_in.__len__())
    sigmoid_scores_ind = np.zeros(test_loader_in.batch_size*test_loader_in.__len__())
    for index, data in enumerate(test_loader_in):

        images, _ = data
        images = images.to(device)
        o, _, _ = _process_gen_odin(model, images, best_epsilon_o)
        _, h, _ = _process_gen_odin(model, images, best_epsilon_h)
        _, _, g = _process_gen_odin(model, images, best_epsilon_g)

        max_o_scores_ind[index*test_loader_in.batch_size:index*test_loader_in.batch_size + o.shape[0]] = o
        max_h_scores_ind[index*test_loader_in.batch_size:index*test_loader_in.batch_size + o.shape[0]] = h
        sigmoid_scores_ind[index*test_loader_in.batch_size:index*test_loader_in.batch_size + o.shape[0]] = np.squeeze(g)
        len_ += o.shape[0]

    max_o_scores_ind = max_o_scores_ind[:len_]
    max_h_scores_ind = max_h_scores_ind[:len_]
    sigmoid_scores_ind = sigmoid_scores_ind[:len_]

    len_ = 0
    max_o_scores_ood = np.zeros(ood_loader.batch_size*ood_loader.__len__())
    max_h_scores_ood = np.zeros(ood_loader.batch_size*ood_loader.__len__())
    sigmoid_scores_ood = np.zeros(ood_loader.batch_size*ood_loader.__len__())
    for index, data in enumerate(ood_loader):

        images, _ = data
        images = images.to(device)
        o, _, _ = _process_gen_odin(model, images, best_epsilon_o)
        _, h, _ = _process_gen_odin(model, images, best_epsilon_h)
        _, _, g = _process_gen_odin(model, images, best_epsilon_g)

        max_o_scores_ood[index*ood_loader.batch_size:index*ood_loader.batch_size + o.shape[0]] = o
        max_h_scores_ood[index*ood_loader.batch_size:index*ood_loader.batch_size + o.shape[0]] = h
        sigmoid_scores_ood[index*ood_loader.batch_size:index*ood_loader.batch_size + o.shape[0]] = np.squeeze(g)
        len_ += o.shape[0]

    max_o_scores_ood = max_o_scores_ood[:len_]
    max_h_scores_ood = max_h_scores_ood[:len_]
    sigmoid_scores_ood = sigmoid_scores_ood[:len_]

    if exclude_class is None:
        max_o_ind_savefile_name = f'npzs/max_o_gen_odin_{mode}_{ind_dataset}.npz'
        max_h_ind_savefile_name = f'npzs/max_h_gen_odin_{mode}_{ind_dataset}.npz'
        sigmoid_ind_savefile_name = f'npzs/sigmoid_gen_odin_{mode}_{ood_dataset}.npz'
        max_o_ood_savefile_name = f'npzs/max_o_gen_odin_{mode}_{ind_dataset}.npz'
        max_h_ood_savefile_name = f'npzs/max_h_gen_odin_{mode}_{ind_dataset}.npz'
        sigmoid_ood_savefile_name = f'npzs/sigmoid_gen_odin_{mode}_{ood_dataset}.npz'
    else:
        max_o_ind_savefile_name = f'npzs/max_o_gen_odin_{mode}_{ind_dataset}_{exclude_class}.npz'
        max_h_ind_savefile_name = f'npzs/max_h_gen_odin_{mode}_{ind_dataset}_{exclude_class}.npz'
        sigmoid_ind_savefile_name = f'npzs/sigmoid_gen_odin_{mode}_{ood_dataset}_{exclude_class}.npz'
        max_o_ood_savefile_name = f'npzs/max_o_gen_odin_{mode}_{ind_dataset}_{exclude_class}.npz'
        max_h_ood_savefile_name = f'npzs/max_h_gen_odin_{mode}_{ind_dataset}_{exclude_class}.npz'
        sigmoid_ood_savefile_name = f'npzs/sigmoid_gen_odin_{mode}_{ood_dataset}_{exclude_class}.npz'

    np.savez(max_o_ind_savefile_name, max_o_scores_ind)
    np.savez(max_h_ind_savefile_name, max_h_scores_ind)
    np.savez(sigmoid_ind_savefile_name, sigmoid_scores_ind)
    np.savez(max_o_ood_savefile_name, max_o_scores_ood)
    np.savez(max_h_ood_savefile_name, max_h_scores_ood)
    np.savez(sigmoid_ood_savefile_name, sigmoid_scores_ood)
    auc_o, fpr_o = _score_npzs(max_o_scores_ind, max_o_scores_ood)
    auc_h, fpr_h = _score_npzs(max_h_scores_ind, max_h_scores_ood)
    auc_sigmoid, fpr_sigmoid = _score_npzs(sigmoid_scores_ind, sigmoid_scores_ood)
    print('###############################################')
    print()
    print(f'Succesfully stored in-distribution ood scores for maximum output to {max_o_ind_savefile_name} and out-distribution ood scores to {max_o_ood_savefile_name}')
    print(f'Succesfully stored in-distribution ood scores for maximum softmax to {max_h_ind_savefile_name} and out-distribution ood scores to {max_h_ood_savefile_name}')
    print(f'Succesfully stored in-distribution sigmoid scores to {sigmoid_ind_savefile_name} and out-distribution ood scores to {sigmoid_ood_savefile_name}')
    print()
    print('###############################################')
    print()
    print(f"Generalized-Odin results (maximum output) on {ind_dataset} (In) vs {ood_dataset} (Out):")
    print(f'Area Under Receiver Operating Characteristic curve: {auc_o}')
    print(f'False Positive Rate @ 95% True Positive Rate: {fpr_o}')
    print('###############################################')
    print()
    print(f"Generalized-Odin results (maximum softmax probability) on {ind_dataset} (In) vs {ood_dataset} (Out):")
    print(f'Area Under Receiver Operating Characteristic curve: {auc_h}')
    print(f'False Positive Rate @ 95% True Positive Rate: {fpr_h}')
    print()
    print('###############################################')
    print()
    print(f"Generalized-Odin results (sigmoid) on {ind_dataset} (In) vs {ood_dataset} (Out):")
    print(f'Area Under Receiver Operating Characteristic curve: {auc_sigmoid}')
    print(f'False Positive Rate @ 95% True Positive Rate: {fpr_sigmoid}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Out-of-Distribution Detection')

    parser.add_argument('--ood_method', '--m', required=True)
    parser.add_argument('--num_classes', '--nc', type=int, required=True)
    parser.add_argument('--in_distribution_dataset', '--in', required=True)
    parser.add_argument('--out_distribution_dataset', '--out', required=True)
    parser.add_argument('--model_checkpoint', '--mc', default=None, required=False)
    parser.add_argument('--model_type', '--mt', default='efficient', required=False)
    parser.add_argument('--model_checkpoints_file', '--mcf', default=None, required=False)
    parser.add_argument('--monte_carlo_steps', '--mcdo', type=int, default=1, required=False)
    parser.add_argument('--temperature', '--T', type=int, default=1, required=False)
    parser.add_argument('--epsilon', '--e', type=float, default=0, required=False)
    parser.add_argument('--with_FGSM', '--fgsm', type=bool, default=False, required=False)
    parser.add_argument('--batch_size', '--bs', type=int, default=32, required=False)
    parser.add_argument('--exclude_class', '--ex', default=None, required=False)
    parser.add_argument('--gen_odin_mode', '--gom', type=int, default=0, required=False)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)
    parser.add_argument('--score_ind', '--sind', type=int, default=1, required=False)
    parser.add_argument('--scaling', '--sc', type=int, default=1, required=False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')
    score_ind = True if args.score_ind == 1 else False

    ood_method = args.ood_method.lower()

    if args.model_checkpoint is None and args.model_checkpoints_file is None:
        raise NotImplementedError('You need to specify either a single or multiple checkpoints')
    elif args.model_checkpoint is not None:
        if args.model_type == 'efficient':
            if ood_method == 'self-supervision' or ood_method == 'selfsupervision' or ood_method =='self_supervision' or ood_method =='rotation':
                model = build_model_with_checkpoint('roteb0', args.model_checkpoint, device=device, out_classes=args.num_classes)
            elif ood_method == 'generalized-odin' or ood_method == 'generalizedodin':
                model = build_model_with_checkpoint('geneb0', args.model_checkpoint, device=device, out_classes=args.num_classes)
            else:
                model = build_model_with_checkpoint('eb0', args.model_checkpoint, device=device, out_classes=args.num_classes)
        elif args.model_type == 'wide':
            model = build_model_with_checkpoint('wide', args.model_checkpoint, device=device, out_classes=args.num_classes)
        elif args.model_type == 'dense':
            if ood_method == 'generalized-odin' or ood_method == 'generalizedodin':
                model = build_model_with_checkpoint('genDense', args.model_checkpoint, device=device, out_classes=args.num_classes, gen_odin_mode=args.gen_odin_mode)
            else:
                raise NotImplementedError('DenseNet can only ne used with Generalized-Odin at the monent')
    else:
        model_checkpoints = []
        for line in open(args.model_checkpoints_file, 'r'):
            model_checkpoints.append(line.split('\n')[0])

    # standard_datasets = ['cifar10']
    # if args.in_distribution_dataset.lower() == 'cifar10' and args.out_distribution_dataset.lower() == 'tinyimagenet':
    #     pickle_files = ['train_indices_cifar10.pickle', 'val_indices_cifar10.pickle']
    #     if args.model_type == 'efficient':
    #         trainloader, val_loader, testloader = cifar10loaders(train_batch_size=args.batch_size, test_batch_size=args.batch_size, pickle_files=pickle_files, test=True)
    #         ood_loader = tinyImageNetloader(batch_size=args.batch_size)
    #     else:
    #         trainloader, val_loader, testloader = cifar10loaders(train_batch_size=args.batch_size, test_batch_size=args.batch_size, pickle_files=pickle_files, test=True, resize=False)
    #         ood_loader = tinyImageNetloader(batch_size=args.batch_size, resize=False)

    testloader, ood_loader = get_temp_data_loaders()
    # if args.with_FGSM or ood_method == 'generalized-odin' or ood_method == 'generalizedodin':
    #     loaders = [val_loader, testloader, ood_loader]
    # else:
    loaders = [testloader, ood_loader]

    if ood_method == 'baseline':
        if args.with_FGSM:
            print('FGSM cannot be combined with the baseline method, skipping this step')
        _baseline(model, loaders, ind_dataset=args.in_distribution_dataset, ood_dataset=args.out_distribution_dataset, monte_carlo_steps=args.monte_carlo_steps, exclude_class=args.exclude_class, device=device, score_ind=score_ind)

    elif ood_method == 'odin':
        if args.temperature != 1 or args.epsilon != 0:
            _odin(model, loaders, ind_dataset=args.in_distribution_dataset, ood_dataset=args.out_distribution_dataset, T=args.temperature, epsilon=args.epsilon, with_fgsm=args.with_FGSM, exclude_class=args.exclude_class, device=device, score_ind=score_ind)
        else:
            _odin(model, loaders, ind_dataset=args.in_distribution_dataset, ood_dataset=args.out_distribution_dataset, with_fgsm=args.with_FGSM, exclude_class=args.exclude_class, device=device, score_ind=score_ind)

    elif ood_method == 'mahalanobis':
        if not args.with_FGSM:
            print('Mahalanobis method can only be used with FGSM, applying it either way')
        loaders = [trainloader] + [val_loader, testloader, ood_loader]
        _generate_Mahalanobis(model, loaders=loaders, ind_dataset=args.in_distribution_dataset, ood_dataset=args.out_distribution_dataset, num_classes=args.num_classes, exclude_class=args.exclude_class, device=device, score_ind=score_ind)

    elif ood_method == 'self-supervision' or ood_method =='selfsupervision' or ood_method =='self_supervision' or ood_method =='rotation':
        if args.with_FGSM:
            print('FGSM cannot be combined with the self-supervision method, skipping this step')
        _rotation(model, loaders, ind_dataset=args.in_distribution_dataset, ood_dataset=args.out_distribution_dataset, num_classes=args.num_classes, exclude_class=args.exclude_class, device=device)

    elif ood_method == 'ensemble':
        if args.with_FGSM:
            print('FGSM cannot be combined with the ensemble method, skipping this step')
        scaling = True if args.scaling == 1 else False
        _ensemble_inference(model_checkpoints, loaders, out_classes=args.num_classes, ind_dataset=args.in_distribution_dataset, ood_dataset=args.out_distribution_dataset, mode=args.ensemble_mode, device=device, scaling=scaling)

    elif ood_method == 'generalized-odin' or ood_method == 'generalizedodin':
        # _gen_odin_inference(model, loaders, ind_dataset=args.in_distribution_dataset, ood_dataset=args.out_distribution_dataset, mode=args.gen_odin_mode, exclude_class=args.exclude_class)
        _gen_odin_temp(model, loaders, ind_dataset=args.in_distribution_dataset, device=device)
    else:
        raise NotImplementedError('Requested unknown Out-of-Distribution Detection Method')
