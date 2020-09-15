import torch
import numpy as np
from torch.autograd import Variable
from utils import build_model_with_checkpoint
from dataLoaders import get_ood_loaders, get_triplets_loaders
from tqdm import tqdm
import lib_generation
import argparse
import os
import random
from ood import _get_metrics, _score_npzs, _find_threshold, _get_odin_scores, _score_mahalanobis
from ood_ensemble import _ood_detection_performance
import ipdb

abs_path = '/home/ferles/Dermatology/medusa/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _odin(model, loaders, device, ood_dataset_1, ood_dataset_2, ood_dataset_3):

    model.eval()
    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders

    T = 1000

    for epsilon in tqdm([-0.1, -0.0005, 0, 0.0005, 0.1]):

        val_ind = _get_odin_scores(model, val_ind_loader, T, epsilon, device=device)
        val_ood = _get_odin_scores(model, val_ood_loader, T, epsilon, device=device)

        ind = _get_odin_scores(model, test_ind_loader, T, epsilon, device=device)
        ood_1 = _get_odin_scores(model, test_ood_loader_1, T, epsilon, device=device)
        ood_2 = _get_odin_scores(model, test_ood_loader_2, T, epsilon, device=device)
        ood_3 = _get_odin_scores(model, test_ood_loader_3, T, epsilon, device=device)

        print(f'########## epsilon: {epsilon} ##########')
        _ood_detection_performance('Odin', val_ind, val_ood, ind, ood_1, ood_2, ood_3, ood_dataset_1, ood_dataset_2, ood_dataset_3)


def _generate_Mahalanobis(model, loaders, device, ood_dataset_1, ood_dataset_2, ood_dataset_3, num_classes, model_type='eb0'):

    model.eval()
    train_ind_loader, val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders

    temp_x = torch.rand(2, 3, 224, 224).to(device)
    temp_x = Variable(temp_x)
    temp_x = temp_x.to(device)
    if model_type == 'eb0':
        idxs = [0, 2, 4, 7, 10, 14, 15]
        x, features = model.extract_features(temp_x, mode='all')
    features = [features[idx] for idx in idxs] + [x]
    num_output = len(features)
    feature_list = np.empty(num_output)
    count = 0
    for out in features:
        feature_list[count] = out.size(1)
        count += 1

    sample_mean, precision = lib_generation.sample_estimator(model, num_classes, feature_list, train_ind_loader, device=device)

    m_list = [-0.01, -0.0005, 0.0, 0.0005, 0.01]
    for magnitude in tqdm(m_list):

        for i in range(num_output):
            M_val = lib_generation.get_Mahalanobis_score(model, val_ind_loader, num_classes, sample_mean, precision, i, magnitude, device=device)
            M_val = np.asarray(M_val, dtype=np.float32)
            if i == 0:
                Mahalanobis_val_ind = M_val.reshape((M_val.shape[0], -1))
            else:
                Mahalanobis_val_ind = np.concatenate((Mahalanobis_val_ind, M_val.reshape((M_val.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_val_ood = lib_generation.get_Mahalanobis_score(model, val_ood_loader, num_classes, sample_mean, precision, i, magnitude, device=device)
            M_val_ood = np.asarray(M_val_ood, dtype=np.float32)
            if i == 0:
                Mahalanobis_val_ood = M_val_ood.reshape((M_val_ood.shape[0], -1))
            else:
                Mahalanobis_val_ood = np.concatenate((Mahalanobis_val_ood, M_val_ood.reshape((M_val_ood.shape[0], -1))), axis=1)

        Mahalanobis_val_ind = np.asarray(Mahalanobis_val_ind, dtype=np.float32)
        Mahalanobis_val_ood = np.asarray(Mahalanobis_val_ood, dtype=np.float32)

        regressor, _, _ = _score_mahalanobis(Mahalanobis_val_ind, Mahalanobis_val_ood)

        val_ind = regressor.predict_proba(Mahalanobis_val_ind)[:, 1]
        val_ood = regressor.predict_proba(Mahalanobis_val_ood)[:, 1]

        for i in range(num_output):
            M_test = lib_generation.get_Mahalanobis_score(model, test_ind_loader, num_classes, sample_mean, precision, i, magnitude, device=device)
            M_test = np.asarray(M_test, dtype=np.float32)
            if i == 0:
                Mahalanobis_test = M_test.reshape((M_test.shape[0], -1))
            else:
                Mahalanobis_test = np.concatenate((Mahalanobis_test, M_test.reshape((M_test.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_ood_1 = lib_generation.get_Mahalanobis_score(model, test_ood_loader_1, num_classes, sample_mean, precision, i, magnitude, device=device)
            M_ood_1 = np.asarray(M_ood_1, dtype=np.float32)
            if i == 0:
                Mahalanobis_ood_1 = M_ood_1.reshape((M_ood_1.shape[0], -1))
            else:
                Mahalanobis_ood_1 = np.concatenate((Mahalanobis_ood_1, M_ood_1.reshape((M_ood_1.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_ood_2 = lib_generation.get_Mahalanobis_score(model, test_ood_loader_2, num_classes, sample_mean, precision, i, magnitude, device=device)
            M_ood_2 = np.asarray(M_ood_2, dtype=np.float32)
            if i == 0:
                Mahalanobis_ood_2 = M_ood_2.reshape((M_ood_2.shape[0], -2))
            else:
                Mahalanobis_ood_2 = np.concatenate((Mahalanobis_ood_2, M_ood_2.reshape((M_ood_2.shape[0], -2))), axis=1)

        for i in range(num_output):
            M_ood_3 = lib_generation.get_Mahalanobis_score(model, test_ood_loader_3, num_classes, sample_mean, precision, i, magnitude, device=device)
            M_ood_3 = np.asarray(M_ood_3, dtype=np.float32)
            if i == 0:
                Mahalanobis_ood_3 = M_ood_3.reshape((M_ood_3.shape[0], -3))
            else:
                Mahalanobis_ood_3 = np.concatenate((Mahalanobis_ood_3, M_ood_3.reshape((M_ood_3.shape[0], -3))), axis=1)

        Mahalanobis_test = np.asarray(Mahalanobis_test, dtype=np.float32)
        Mahalanobis_ood_1 = np.asarray(Mahalanobis_ood_1, dtype=np.float32)
        Mahalanobis_ood_2 = np.asarray(Mahalanobis_ood_2, dtype=np.float32)
        Mahalanobis_ood_3 = np.asarray(Mahalanobis_ood_3, dtype=np.float32)

        ind = regressor.predict_proba(Mahalanobis_test)[:, 1]
        ood_1 = regressor.predict_proba(Mahalanobis_ood_1)[:, 1]
        ood_2 = regressor.predict_proba(Mahalanobis_ood_2)[:, 1]
        ood_3 = regressor.predict_proba(Mahalanobis_ood_3)[:, 1]

        print(f'########## epsilon: {magnitude} ##########')
        _ood_detection_performance('Mahalanobis', val_ind, val_ood, ind, ood_1, ood_2, ood_3, ood_dataset_1, ood_dataset_2, ood_dataset_3)


if __name__ == '__main__':

    import time
    start = time.time()

    parser = argparse.ArgumentParser(description='Out-of-Distribution Detection')

    parser.add_argument('--in_distribution_dataset', '--in', required=True)
    parser.add_argument('--val_dataset', '--val', required=True)
    parser.add_argument('--test_dataset', '--test', default='tinyimagenet', required=False)
    parser.add_argument('--num_classes', '--nc', type=int, required=True)
    parser.add_argument('--model_checkpoint', '--mc', default=None, required=False)
    parser.add_argument('--batch_size', '--bs', type=int, default=32, required=False)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')

    standard_checkpoint = args.model_checkpoint
    standard_model = build_model_with_checkpoint('eb0', standard_checkpoint, device=device, out_classes=args.num_classes)

    ind_dataset = args.in_distribution_dataset.lower()
    val_dataset = args.val_dataset.lower()
    all_datasets = ['cifar10', 'cifar100', 'svhn', 'stl', args.test_dataset]
    all_datasets.remove(ind_dataset)
    all_datasets.remove(val_dataset)
    ood_dataset_1, ood_dataset_2, ood_dataset_3 = all_datasets

    loaders = get_triplets_loaders(batch_size=args.batch_size, ind_dataset=ind_dataset, val_ood_dataset=val_dataset, ood_datasets=all_datasets)
    mahalanobis_loaders = get_triplets_loaders(batch_size=20, ind_dataset=ind_dataset, val_ood_dataset=val_dataset, ood_datasets=all_datasets)

    _odin(standard_model, loaders[1:], device, ood_dataset_1, ood_dataset_2, ood_dataset_3)
    _generate_Mahalanobis(standard_model, mahalanobis_loaders, device, ood_dataset_1, ood_dataset_2, ood_dataset_3, num_classes=args.num_classes)