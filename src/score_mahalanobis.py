import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
from utils import build_model_with_checkpoint
from dataLoaders import cifar10loaders, cifar100loaders, tinyImageNetloader, _get_isic_loaders_ood, _get_7point_loaders_ood
from tqdm import tqdm
import lib_generation
import argparse
import os
import random
import pickle


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Out-of-Distribution Detection')
    parser.add_argument('--exclude_class', '--ex', default=None, required=False)
    args = parser.parse_args()

    exclude_class = args.exclude_class

    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    # if exclude_class !='AK' and exclude_class !='SCC':
    #
    #     print(f'######################################')
    #     print('7-point-in')
    #     for magnitude in m_list:
    #
    #       with open(f'lr_pickles/logistic_regressor_{exclude_class}_{magnitude}.pickle', 'rb') as lrp:
    #             lr = pickle.load(lrp)
    #
    #             ind = np.load(f'npzs/Mahalanobis_ISIC_{magnitude}_{exclude_class}.npz')['arr_0']
    #             ood = np.load(f'npzs/Mahalanobis_7-point-in_{magnitude}_{exclude_class}.npz')['arr_0']
    #
    #             y_known = np.ones(ind.shape[0])
    #             y_novel = np.zeros(ood.shape[0])
    #
    #             X = np.append(ind, ood, axis=0)
    #             y = np.append(y_known, y_novel)
    #
    #             known_preds = lr.predict_proba(ind)[:, 1]
    #             novel_preds = lr.predict_proba(ood)[:, 1]
    #
    #             fpr, tpr, _ = roc_curve(y, np.append(known_preds, novel_preds, axis=0))
    #             roc_auc = round(100*auc(fpr, tpr), 2)
    #
    #             perc = 0.95
    #             sorted_ = np.sort(known_preds)
    #             threshold = sorted_[int((1-perc)*known_preds.shape[0])]
    #
    #             ind_ = np.zeros(known_preds.shape)
    #             ood_ = np.zeros(novel_preds.shape)
    #             ind_[np.argwhere(known_preds > threshold)] = 1
    #             ood_[np.argwhere(novel_preds > threshold)] = 1
    #
    #             X = np.append(ind_, ood_)
    #             bool_X = np.atleast_1d(X.astype(np.bool))
    #             bool_y = np.atleast_1d(y.astype(np.bool))
    #
    #             tn = np.count_nonzero(~bool_X & ~bool_y)
    #             fp = np.count_nonzero(bool_X & ~bool_y)
    #
    #             fpr = round(100*fp/(fp+tn), 2)
    #
    #             print()
    #             print(f'{exclude_class}')
    #             print(f'{magnitude}')
    #             print(f'AUC: {roc_auc}')
    #             print(f'FPR: {fpr}')
    #             print()
    #
    # print()
    # print(f'######################################')
    # print('7-point-out')
    # for magnitude in m_list:
    #
    #     with open(f'lr_pickles/logistic_regressor_{exclude_class}_{magnitude}.pickle', 'rb') as lrp:
    #         lr = pickle.load(lrp)
    #
    #     ind = np.load(f'npzs/Mahalanobis_ISIC_{magnitude}_{exclude_class}.npz')['arr_0']
    #     ood = np.load(f'npzs/Mahalanobis_7-point-out_{magnitude}_{exclude_class}.npz')['arr_0']
    #
    #     y_known = np.ones(ind.shape[0])
    #     y_novel = np.zeros(ood.shape[0])
    #
    #     X = np.append(ind, ood, axis=0)
    #     y = np.append(y_known, y_novel)
    #
    #     known_preds = lr.predict_proba(ind)[:, 1]
    #     novel_preds = lr.predict_proba(ood)[:, 1]
    #
    #     fpr, tpr, _ = roc_curve(y, np.append(known_preds, novel_preds, axis=0))
    #     roc_auc = round(100*auc(fpr, tpr), 2)
    #
    #     perc = 0.95
    #     sorted_ = np.sort(known_preds)
    #     threshold = sorted_[int((1-perc)*known_preds.shape[0])]
    #
    #     ind_ = np.zeros(known_preds.shape)
    #     ood_ = np.zeros(novel_preds.shape)
    #     ind_[np.argwhere(known_preds > threshold)] = 1
    #     ood_[np.argwhere(novel_preds > threshold)] = 1
    #
    #     X = np.append(ind_, ood_)
    #     bool_X = np.atleast_1d(X.astype(np.bool))
    #     bool_y = np.atleast_1d(y.astype(np.bool))
    #
    #     tn = np.count_nonzero(~bool_X & ~bool_y)
    #     fp = np.count_nonzero(bool_X & ~bool_y)
    #
    #     fpr = round(100*fp/(fp+tn), 2)
    #
    #     print()
    #     print(f'{exclude_class}')
    #     print(f'{magnitude}')
    #     print(f'AUC: {roc_auc}')
    #     print(f'FPR: {fpr}')
    #     print()

    st = 'Dermofit-out'
    print(f'{st}')
    for magnitude in m_list:

        with open(f'lr_pickles/logistic_regressor_None_{magnitude}.pickle', 'rb') as lrp:
            lr = pickle.load(lrp)

        ind = np.load(f'npzs/Mahalanobis_ISIC_{magnitude}.npz')['arr_0']
        ood = np.load(f'npzs/Mahalanobis_{st}_{magnitude}.npz')['arr_0']

        y_known = np.ones(ind.shape[0])
        y_novel = np.zeros(ood.shape[0])

        X = np.append(ind, ood, axis=0)
        y = np.append(y_known, y_novel)

        known_preds = lr.predict_proba(ind)[:, 1]
        novel_preds = lr.predict_proba(ood)[:, 1]

        fpr, tpr, _ = roc_curve(y, np.append(known_preds, novel_preds, axis=0))
        roc_auc = round(100*auc(fpr, tpr), 2)

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

        fpr = round(100*fp/(fp+tn), 2)

        print()
        print(f'{magnitude}')
        print(f'AUC: {roc_auc}')
        print(f'FPR: {fpr}')
        print()