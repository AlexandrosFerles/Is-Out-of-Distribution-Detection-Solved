import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
from utils import build_model_with_checkpoint
from dataLoaders import get_triplets_loaders
from tqdm import tqdm
import lib_generation
import argparse
import os
import random
import pickle
import ipdb
from ood import _find_threshold, _score_npzs, _score_mahalanobis, _predict_mahalanobis, _get_baseline_scores, _get_odin_scores, _process, _predict_rotations, _process_gen_odin_loader

abs_path = '/home/ferles/Dermatology/medusa/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _verbose(method, ood_dataset_1, ood_dataset_2, ood_dataset_3, aucs, fprs, accs):

    print('###############################################')
    print()
    print(f"{method} results on {ood_dataset_1} (Out):")
    print()
    print(f'Area Under Receiver Operating Characteristic curve: {aucs[0]}')
    print(f'False Positive Rate @ 95% True Positive Rate: {fprs[0]}')
    print(f'Detection Accuracy: {accs[0]}')
    print('###############################################')
    print()
    print(f"{method} results on {ood_dataset_2} (Out):")
    print()
    print(f'Area Under Receiver Operating Characteristic curve: {aucs[1]}')
    print(f'False Positive Rate @ 95% True Positive Rate: {fprs[1]}')
    print(f'Detection Accuracy: {accs[1]}')
    print('###############################################')
    print()
    print(f"{method} results on {ood_dataset_3} (Out):")
    print()
    print(f'Area Under Receiver Operating Characteristic curve: {aucs[2]}')
    print(f'False Positive Rate @ 95% True Positive Rate: {fprs[2]}')
    print(f'Detection Accuracy: {accs[2]}')
    print('###############################################')
    print()
    print('###############################################')
    print('###############################################')
    print(f"MEAN PERFORMANCE OF {method.upper()}:")
    print(f'Area Under Receiver Operating Characteristic curve: {round(np.mean(aucs), 2)} with variance {round(np.std(aucs), 2)}')
    print(f'False Positive Rate @ 95% True Positive Rate: {round(np.mean(fprs), 2)} with variance {round(np.std(fprs), 2)}')
    print(f'Detection Accuracy: {round(np.mean(accs), 2)} with variance {round(np.std(accs), 2)}')
    print('###############################################')
    print('###############################################')


def _baseline(model, loaders, device):

    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders
    model.eval()

    test_ind = _get_baseline_scores(model, test_ind_loader, device, monte_carlo_steps=1)
    test_ood_1 = _get_baseline_scores(model, test_ood_loader_1, device, monte_carlo_steps=1)
    test_ood_2 = _get_baseline_scores(model, test_ood_loader_2, device, monte_carlo_steps=1)
    test_ood_3 = _get_baseline_scores(model, test_ood_loader_3, device, monte_carlo_steps=1)

    return test_ind, test_ood_1, test_ood_2, test_ood_3


def _odin(model, loaders, device):

    model.eval()
    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders

    best_auc = 0

    for T in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        for epsilon in tqdm(np.arange(0, 0.004, 0.004/21, float).tolist()):

            val_ind = _get_odin_scores(model, val_ind_loader, T, epsilon, device=device)
            val_ood = _get_odin_scores(model, val_ood_loader, T, epsilon, device=device)
            auc, _, _ = _score_npzs(val_ind, val_ood, threshold=0)

            if auc > best_auc:
                best_auc = auc
                best_epsilon = epsilon
                best_T = T
                best_val_ind = val_ind
                best_val_ood = val_ood

    print('###############################################')
    print()
    print(f'Selected temperature: {best_T}, selected epsilon: {best_epsilon}')
    print()

    test_ind = _get_odin_scores(model, test_ind_loader, best_T, best_epsilon, device=device)
    test_ood_1 = _get_odin_scores(model, test_ood_loader_1, best_T, best_epsilon, device=device)
    test_ood_2 = _get_odin_scores(model, test_ood_loader_2, best_T, best_epsilon, device=device)
    test_ood_3 = _get_odin_scores(model, test_ood_loader_3, best_T, best_epsilon, device=device)

    return test_ind, test_ood_1, test_ood_2, test_ood_3


def _generate_Mahalanobis(model, loaders, device, num_classes, model_type='eb0'):

    model.eval()
    train_ind_loader, val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders

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

    sample_mean, precision = lib_generation.sample_estimator(model, num_classes, feature_list, train_ind_loader, device=device)

    best_auc = 0
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]

    best_magnitudes, best_fprs, regressors, thresholds = [], [], [], []
    for magnitude in m_list:
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

        regressor, auc, threshold = _score_mahalanobis(Mahalanobis_val_ind, Mahalanobis_val_ood)
        with open(f'lr_pickles/logistic_regressor_{ind_dataset}_{val_dataset}_{magnitude}.pickle', 'wb') as lrp:
            pickle.dump(regressor, lrp, protocol=pickle.HIGHEST_PROTOCOL)

        if auc > best_auc:
            best_auc = auc
            best_magnitudes = [magnitude]
            regressors = [regressor]
            thresholds = [threshold]
        elif auc == best_auc:
            best_magnitudes.append(magnitude)
            regressors.append(regressor)
            thresholds.append(threshold)

    print('###############################################')
    print()
    print(f'Selected magnitudes: {best_magnitudes}')
    print(f'Selected thresholds: {thresholds}')
    print()

    idx = 0
    for (best_magnitude, regressor, threshold) in zip(best_magnitudes, regressors, thresholds):
        for i in range(num_output):
            M_test = lib_generation.get_Mahalanobis_score(model, test_ind_loader, num_classes, sample_mean, precision, i, best_magnitude, device=device)
            M_test = np.asarray(M_test, dtype=np.float32)
            if i == 0:
                Mahalanobis_test = M_test.reshape((M_test.shape[0], -1))
            else:
                Mahalanobis_test = np.concatenate((Mahalanobis_test, M_test.reshape((M_test.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_ood_1 = lib_generation.get_Mahalanobis_score(model, test_ood_loader_1, num_classes, sample_mean, precision, i, best_magnitude, device=device)
            M_ood_1 = np.asarray(M_ood_1, dtype=np.float32)
            if i == 0:
                Mahalanobis_ood_1 = M_ood_1.reshape((M_ood_1.shape[0], -1))
            else:
                Mahalanobis_ood_1 = np.concatenate((Mahalanobis_ood_1, M_ood_1.reshape((M_ood_1.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_ood_2 = lib_generation.get_Mahalanobis_score(model, test_ood_loader_2, num_classes, sample_mean, precision, i, best_magnitude, device=device)
            M_ood_2 = np.asarray(M_ood_2, dtype=np.float32)
            if i == 0:
                Mahalanobis_ood_2 = M_ood_2.reshape((M_ood_2.shape[0], -2))
            else:
                Mahalanobis_ood_2 = np.concatenate((Mahalanobis_ood_2, M_ood_2.reshape((M_ood_2.shape[0], -2))), axis=1)

        for i in range(num_output):
            M_ood_3 = lib_generation.get_Mahalanobis_score(model, test_ood_loader_3, num_classes, sample_mean, precision, i, best_magnitude, device=device)
            M_ood_3 = np.asarray(M_ood_3, dtype=np.float32)
            if i == 0:
                Mahalanobis_ood_3 = M_ood_3.reshape((M_ood_3.shape[0], -3))
            else:
                Mahalanobis_ood_3 = np.concatenate((Mahalanobis_ood_3, M_ood_3.reshape((M_ood_3.shape[0], -3))), axis=1)

            if idx == 0:
                test_ind = regressor.predict_proba(Mahalanobis_test)[:, 1]
                test_ood_1 = regressor.predict_proba(Mahalanobis_ood_1)[:, 1]
                test_ood_2 = regressor.predict_proba(Mahalanobis_ood_1)[:, 2]
                test_ood_3 = regressor.predict_proba(Mahalanobis_ood_1)[:, 3]
            else:
                test_ind += regressor.predict_proba(Mahalanobis_test)[:, 1]
                test_ood_1 += regressor.predict_proba(Mahalanobis_ood_1)[:, 1]
                test_ood_2 += regressor.predict_proba(Mahalanobis_ood_1)[:, 2]
                test_ood_3 += regressor.predict_proba(Mahalanobis_ood_1)[:, 3]
            idx += 1

        test_ind /= idx
        test_ood_1 /= idx
        test_ood_2 /= idx
        test_ood_3 /= idx

    return test_ind, test_ood_1, test_ood_2, test_ood_3


def _rotation(model, loaders, device, num_classes):

    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders

    _, _, val_ind_full = _predict_rotations(model, val_ind_loader, num_classes, device=device)
    _, _, val_ood_full = _predict_rotations(model, val_ood_loader, num_classes, device=device)

    _, threshold = _find_threshold(val_ind_full, val_ood_full)

    _, _, ind_full = _predict_rotations(model, test_ind_loader, num_classes, device=device)
    _, _, ood_full_1 = _predict_rotations(model, test_ood_loader_1, num_classes, device=device)
    _, _, ood_full_2 = _predict_rotations(model, test_ood_loader_2, num_classes, device=device)
    _, _, ood_full_3 = _predict_rotations(model, test_ood_loader_3, num_classes, device=device)

    return ind_full, ood_full_1, ood_full_2, ood_full_3


def _gen_odin_inference(model, loaders, device):

    model.eval()
    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders

    epsilons = [0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
    best_auc, best_epsilon = 0, 0

    for epsilon in epsilons:

        val_ind_scores = _process_gen_odin_loader(model, val_ind_loader, device, epsilon)
        val_ood_scores = _process_gen_odin_loader(model, val_ood_loader, device, epsilon)

        auc, _, _ = _score_npzs(val_ind_scores, val_ood_scores, threshold=0)
        if auc > best_auc:
            best_auc = auc
            best_epsilon = epsilon
            best_val_ind_scores = val_ind_scores
            best_val_ood_scores = val_ood_scores

    test_ind_scores = _process_gen_odin_loader(model, test_ind_loader, device, best_epsilon)
    test_ood_scores_1 = _process_gen_odin_loader(model, test_ood_loader_1, device, best_epsilon)
    test_ood_scores_2 = _process_gen_odin_loader(model, test_ood_loader_2, device, best_epsilon)
    test_ood_scores_3 = _process_gen_odin_loader(model, test_ood_loader_3, device, best_epsilon)

    return test_ind_scores, test_ood_scores_1, test_ood_scores_2, test_ood_scores_3


def _ensemble_inference(model_checkpoints, num_classes, loaders, device, ind_dataset, val_dataset, T=1000, epsilon=0.002, scaling=True):

    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders

    index = 0
    for model_checkpoint in tqdm(model_checkpoints):
        model = build_model_with_checkpoint('eb0', model_checkpoint, device, out_classes=num_classes[index])
        model.eval()
        if scaling:
            if index == 0:
                test_ind = _get_odin_scores(model, test_ind_loader, T, epsilon, device=device, score_entropy=True)
                test_ood_1 = _get_odin_scores(model, test_ood_loader_1, T, epsilon, device=device, score_entropy=True)
                test_ood_2 = _get_odin_scores(model, test_ood_loader_2, T, epsilon, device=device, score_entropy=True)
                test_ood_3 = _get_odin_scores(model, test_ood_loader_3, T, epsilon, device=device, score_entropy=True)
            else:
                test_ind += _get_odin_scores(model, test_ind_loader, T, epsilon, device=device, score_entropy=True)
                test_ood_1 += _get_odin_scores(model, test_ood_loader_1, T, epsilon, device=device, score_entropy=True)
                test_ood_2 += _get_odin_scores(model, test_ood_loader_2, T, epsilon, device=device, score_entropy=True)
                test_ood_3 += _get_odin_scores(model, test_ood_loader_3, T, epsilon, device=device, score_entropy=True)
        else:
            if index == 0:
                test_ind = _get_odin_scores(model, test_ind_loader, T=1, epsilon=0, device=device, score_entropy=True)
                test_ood_1 = _get_odin_scores(model, test_ood_loader_1, T=1, epsilon=0, device=device, score_entropy=True)
                test_ood_2 = _get_odin_scores(model, test_ood_loader_2, T=1, epsilon=0, device=device, score_entropy=True)
                test_ood_3 = _get_odin_scores(model, test_ood_loader_3, T=1, epsilon=0, device=device, score_entropy=True)
            else:
                test_ind += _get_odin_scores(model, test_ind_loader, T=1, epsilon=0, device=device, score_entropy=True)
                test_ood_1 += _get_odin_scores(model, test_ood_loader_1, T=1, epsilon=0, device=device, score_entropy=True)
                test_ood_2 += _get_odin_scores(model, test_ood_loader_2, T=1, epsilon=0, device=device, score_entropy=True)
                test_ood_3 += _get_odin_scores(model, test_ood_loader_3, T=1, epsilon=0, device=device, score_entropy=True)
        index += 1

    test_ind = test_ind / index
    test_ood_1 = test_ood_1 / index
    test_ood_2 = test_ood_2 / index
    test_ood_3 = test_ood_3 / index

    return test_ind, test_ood_1, test_ood_2, test_ood_3


def _update_scores(test_ind, temp_ind, test_ood_1, temp_ood_1, test_ood_2, temp_ood_2, test_ood_3, temp_ood_3):

    test_ind = np.append(test_ind, temp_ind, axis=1)
    test_ood_1 = np.append(test_ood_1, temp_ood_1, axis=1)
    test_ood_2 = np.append(test_ood_2, temp_ood_2, axis=1)
    test_ood_3 = np.append(test_ood_3, temp_ood_3, axis=1)

    return test_ind, test_ood_1, test_ood_2, test_ood_3


if __name__ == '__main__':

    import time
    start = time.time()

    parser = argparse.ArgumentParser(description='Out-of-Distribution Detection')

    parser.add_argument('--in_distribution_dataset', '--in', required=True)
    parser.add_argument('--val_dataset', '--val', required=True)
    parser.add_argument('--num_classes', '--nc', type=int, required=True)
    parser.add_argument('--model_checkpoints_file', '--mcf', default=None, required=False)
    parser.add_argument('--batch_size', '--bs', type=int, default=32, required=False)
    parser.add_argument('--scaling', '--sc', type=bool, default=True, required=False)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')

    model_checkpoints, num_classes = [], []
    for line in open(args.model_checkpoints_file, 'r'):
        model_checkpoint = line.split('\n')[0]
        model_checkpoints.append(model_checkpoint)

    ipdb.set_trace()

    standard_checkpoint = model_checkpoints[0]
    standard_model = build_model_with_checkpoint('eb0', standard_checkpoint, device=device, out_classes=args.num_classes)

    rotation_checkpoint = model_checkpoints[1]
    rotation_model = build_model_with_checkpoint('roteb0', rotation_checkpoint, device=device, out_classes=args.num_classes)

    genodin_checkpoint = model_checkpoints[2]
    genodin_model = build_model_with_checkpoint('geneb0', genodin_checkpoint, device=device, out_classes=args.num_classes)

    ensemble_checkpoints_file = model_checkpoints[3]

    ensemble_checkpoints, num_classes = [], []
    for line in open(ensemble_checkpoints_file, 'r'):
        model_checkpoint, nc = line.split('\n')[0].split(',')
        nc = int(nc)
        ensemble_checkpoints.append(model_checkpoint)
        num_classes.append(nc)

    ind_dataset = args.in_distribution_dataset.lower()
    val_dataset = args.val_dataset.lower()
    all_datasets = ['cifar10', 'cifar100', 'svhn', 'stl', 'tinyimagenet']
    all_datasets.remove(ind_dataset)
    all_datasets.remove(val_dataset)
    ood_dataset_1, ood_dataset_2, ood_dataset_3 = all_datasets

    loaders = get_triplets_loaders(batch_size=args.batch_size, ind_dataset=ind_dataset, val_ood_dataset=val_dataset, ood_datasets=all_datasets)

    method_loaders = loaders[1:]
    test_ind, test_ood_1, test_ood_2, test_ood_3 = _baseline(standard_model, method_loaders, device)

    temp_ind, temp_ood_1, temp_ood_2, temp_ood_3 = _rotation(rotation_model, method_loaders, device, num_classes=args.num_classes)
    test_ind, test_ood_1, test_ood_2, test_ood_3 = _update_scores(test_ind, temp_ind, test_ood_1, temp_ood_1, test_ood_2, temp_ood_2, test_ood_3, temp_ood_3)

    # if ood_method == 'odin':
    #     method_loaders = loaders[1:]
    #     _odin(model, method_loaders, device, ind_dataset=ind_dataset, val_dataset=val_dataset, ood_datasets=all_datasets)
    # elif ood_method == 'mahalanobis':
    #     _generate_Mahalanobis(model, loaders, device, ind_dataset=ind_dataset, val_dataset=val_dataset, ood_datasets=all_datasets, num_classes=args.num_classes)
    # elif ood_method == 'generalized-odin' or ood_method == 'generalizedodin':
    #     method_loaders = loaders[1:]
    #     _gen_odin_inference(model, method_loaders, device, ind_dataset=ind_dataset, val_dataset=val_dataset, ood_datasets=all_datasets)
    # elif ood_method == 'ensemble':
    #     method_loaders = loaders[1:]
    #     _ensemble_inference(model_checkpoints, num_classes, method_loaders, device, ind_dataset=args.in_distribution_dataset, val_dataset=args.val_dataset, scaling=args.scaling)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
