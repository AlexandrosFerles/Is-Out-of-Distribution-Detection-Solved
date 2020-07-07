import torch
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from torch.autograd import Variable
from utils import build_model_with_checkpoint
from dataLoaders import get_triplets_loaders
from tqdm import tqdm
import lib_generation
import argparse
import os
import random
import pickle
from ood_triplets import _verbose
from ood import _find_threshold, _score_npzs, _score_mahalanobis, _predict_mahalanobis, _get_baseline_scores, _get_odin_scores, _process, _predict_rotations, _process_gen_odin_loader
import ipdb

abs_path = '/home/ferles/Dermatology/medusa/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _baseline(model, loaders, device):

    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders
    model.eval()

    val_ind = _get_baseline_scores(model, val_ind_loader, device, monte_carlo_steps=1)
    val_ood = _get_baseline_scores(model, val_ood_loader, device, monte_carlo_steps=1)
    test_ind = _get_baseline_scores(model, test_ind_loader, device, monte_carlo_steps=1)
    test_ood_1 = _get_baseline_scores(model, test_ood_loader_1, device, monte_carlo_steps=1)
    test_ood_2 = _get_baseline_scores(model, test_ood_loader_2, device, monte_carlo_steps=1)
    test_ood_3 = _get_baseline_scores(model, test_ood_loader_3, device, monte_carlo_steps=1)

    return val_ind, val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3


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

    return best_val_ind, best_val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3


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
            best_val_ind = regressor.predict_proba(Mahalanobis_val_ind)[:, 1]
            best_val_ood = regressor.predict_proba(Mahalanobis_val_ood)[:, 1]
            cnt = 1
        elif auc == best_auc:
            best_magnitudes.append(magnitude)
            regressors.append(regressor)
            thresholds.append(threshold)
            best_val_ind += regressor.predict_proba(Mahalanobis_val_ind)[:, 1]
            best_val_ood += regressor.predict_proba(Mahalanobis_val_ood)[:, 1]
            cnt += 1

    best_val_ind /= cnt
    best_val_ood /= cnt

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
            test_ood_2 = regressor.predict_proba(Mahalanobis_ood_2)[:, 1]
            test_ood_3 = regressor.predict_proba(Mahalanobis_ood_3)[:, 1]
        else:
            test_ind += regressor.predict_proba(Mahalanobis_test)[:, 1]
            test_ood_1 += regressor.predict_proba(Mahalanobis_ood_1)[:, 1]
            test_ood_2 += regressor.predict_proba(Mahalanobis_ood_2)[:, 1]
            test_ood_3 += regressor.predict_proba(Mahalanobis_ood_3)[:, 1]
        idx += 1

    test_ind /= idx
    test_ood_1 /= idx
    test_ood_2 /= idx
    test_ood_3 /= idx

    return best_val_ind, best_val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3


def _rotation(model, loaders, device, num_classes):

    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders

    val_kl_div_ind, val_rot_score_ind, _ = _predict_rotations(model, val_ind_loader, num_classes, device=device)
    val_kl_div_ood, val_rot_score_ood, _ = _predict_rotations(model, val_ood_loader, num_classes, device=device)

    best_val_auc, best_lamda = 0, 0.25
    for lamda in [0.25, 0.5, 0.75, 1]:

        anomaly_score_ind = val_kl_div_ind - lamda * val_rot_score_ind
        anomaly_score_ood = val_kl_div_ood - lamda * val_rot_score_ood
        auc, _, _ = _score_npzs(anomaly_score_ind, anomaly_score_ood)
        if auc > best_val_auc:
            best_val_auc = auc
            best_lamda = lamda
            best_anomaly_score_ind = anomaly_score_ind
            best_anomaly_score_ood = anomaly_score_ood

    print(f"Chosen lambda: {best_lamda}")
    _, threshold = _find_threshold(best_anomaly_score_ind, best_anomaly_score_ood)

    _, _, ind_full = _predict_rotations(model, test_ind_loader, num_classes, lamda=best_lamda, device=device)
    _, _, ood_full_1 = _predict_rotations(model, test_ood_loader_1, num_classes, lamda=best_lamda, device=device)
    _, _, ood_full_2 = _predict_rotations(model, test_ood_loader_2, num_classes, lamda=best_lamda, device=device)
    _, _, ood_full_3 = _predict_rotations(model, test_ood_loader_3, num_classes, lamda=best_lamda, device=device)

    return best_anomaly_score_ind, best_anomaly_score_ood, ind_full, ood_full_1, ood_full_2, ood_full_3


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

    return best_val_ind_scores, best_val_ood_scores, test_ind_scores, test_ood_scores_1, test_ood_scores_2, test_ood_scores_3


def _ensemble_inference(model_checkpoints, num_classes, loaders, device):

    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders

    models = []
    for index, model_checkpoint in enumerate(model_checkpoints):
        model = build_model_with_checkpoint('eb0', model_checkpoint, device, out_classes=num_classes[index])
        model.eval()
        models.append(model)

    best_auc = -1e30
    for T in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        for epsilon in tqdm(np.arange(0, 0.004, 0.004/21, float).tolist()):
            for index, model in enumerate(models):
                if index == 0:
                    val_ind = _get_odin_scores(model, val_ind_loader, T, epsilon, device=device, score_entropy=True)
                    val_ood = _get_odin_scores(model, val_ood_loader, T, epsilon, device=device, score_entropy=True)
                else:
                    val_ind += _get_odin_scores(model, val_ind_loader, T, epsilon, device=device, score_entropy=True)
                    val_ood += _get_odin_scores(model, val_ood_loader, T, epsilon, device=device, score_entropy=True)

            val_ind = val_ind / len(models)
            val_ood = val_ood / len(models)
            auc, _, _ = _score_npzs(val_ind, val_ood)
            if auc > best_auc:
                best_auc = auc
                best_T, best_epsilon = T, epsilon
                best_val_ind, best_val_ood = val_ind, val_ood

    print(f"Chosen T: {best_T}, epsilon: {best_epsilon}")
    _, threshold = _find_threshold(best_val_ind, best_val_ood)

    for index, model in enumerate(models):
        if index == 0:
            test_ind = _get_odin_scores(model, test_ind_loader, best_T, best_epsilon, device=device, score_entropy=True)
            test_ood_1 = _get_odin_scores(model, test_ood_loader_1, best_T, best_epsilon, device=device, score_entropy=True)
            test_ood_2 = _get_odin_scores(model, test_ood_loader_2, best_T, best_epsilon, device=device, score_entropy=True)
            test_ood_3 = _get_odin_scores(model, test_ood_loader_3, best_T, best_epsilon, device=device, score_entropy=True)
        else:
            test_ind += _get_odin_scores(model, test_ind_loader, best_T, best_epsilon, device=device, score_entropy=True)
            test_ood_1 += _get_odin_scores(model, test_ood_loader_1, best_T, best_epsilon, device=device, score_entropy=True)
            test_ood_2 += _get_odin_scores(model, test_ood_loader_2, best_T, best_epsilon, device=device, score_entropy=True)
            test_ood_3 += _get_odin_scores(model, test_ood_loader_3, best_T, best_epsilon, device=device, score_entropy=True)

    test_ind = test_ind / len(models)
    test_ood_1 = test_ood_1 / len(models)
    test_ood_2 = test_ood_2 / len(models)
    test_ood_3 = test_ood_3 / len(models)

    return best_val_ind, best_val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3


def _update_scores(val_ind, temp_val_ind, val_ood, temp_val_ood, test_ind, temp_ind, test_ood_1, temp_ood_1, test_ood_2, temp_ood_2, test_ood_3, temp_ood_3, expand=False):

    if expand:
        val_ind = np.append(np.expand_dims(val_ind, axis=1), np.expand_dims(temp_val_ind, axis=1), axis=1)
        val_ood = np.append(np.expand_dims(val_ood, axis=1), np.expand_dims(temp_val_ood, axis=1), axis=1)
        test_ind = np.append(np.expand_dims(test_ind, axis=1), np.expand_dims(temp_ind, axis=1), axis=1)
        test_ood_1 = np.append(np.expand_dims(test_ood_1, axis=1), np.expand_dims(temp_ood_1, axis=1), axis=1)
        test_ood_2 = np.append(np.expand_dims(test_ood_2, axis=1), np.expand_dims(temp_ood_2, axis=1), axis=1)
        test_ood_3 = np.append(np.expand_dims(test_ood_3, axis=1), np.expand_dims(temp_ood_3, axis=1), axis=1)
    else:
        val_ind = np.append(val_ind, np.expand_dims(temp_val_ind, axis=1), axis=1)
        val_ood = np.append(val_ood, np.expand_dims(temp_val_ood, axis=1), axis=1)
        test_ind = np.append(test_ind, np.expand_dims(temp_ind, axis=1), axis=1)
        test_ood_1 = np.append(test_ood_1, np.expand_dims(temp_ood_1, axis=1), axis=1)
        test_ood_2 = np.append(test_ood_2, np.expand_dims(temp_ood_2, axis=1), axis=1)
        test_ood_3 = np.append(test_ood_3, np.expand_dims(temp_ood_3, axis=1), axis=1)

    return val_ind, val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3


def _ood_detection_performance(method, val_ind, val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3, ood_dataset_1, ood_dataset_2, ood_dataset_3):

    _, threshold = _find_threshold(val_ind, val_ood)

    auc1, fpr1, acc1 = _score_npzs(test_ind, test_ood_1, threshold)
    auc2, fpr2, acc2 = _score_npzs(test_ind, test_ood_2, threshold)
    auc3, fpr3, acc3 = _score_npzs(test_ind, test_ood_3, threshold)

    aucs = [auc1, auc2, auc3]
    fprs = [fpr1, fpr2, fpr3]
    accs = [acc1, acc2, acc3]

    _verbose(method, ood_dataset_1, ood_dataset_2, ood_dataset_3, aucs, fprs, accs)


if __name__ == '__main__':

    import time
    start = time.time()

    parser = argparse.ArgumentParser(description='Out-of-Distribution Detection')

    parser.add_argument('--in_distribution_dataset', '--in', required=True)
    parser.add_argument('--val_dataset', '--val', required=True)
    parser.add_argument('--test_dataset', '--test', default='tinyimagenet', required=False)
    parser.add_argument('--num_classes', '--nc', type=int, required=True)
    parser.add_argument('--model_checkpoints_file', '--mcf', default=None, required=False)
    parser.add_argument('--batch_size', '--bs', type=int, default=32, required=False)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')

    model_checkpoints, num_classes = [], []
    for line in open(args.model_checkpoints_file, 'r'):
        model_checkpoint = line.split('\n')[0]
        model_checkpoints.append(model_checkpoint)

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
    all_datasets = ['cifar10', 'cifar100', 'svhn', 'stl', args.test_dataset]
    all_datasets.remove(ind_dataset)
    all_datasets.remove(val_dataset)
    ood_dataset_1, ood_dataset_2, ood_dataset_3 = all_datasets

    loaders = get_triplets_loaders(batch_size=args.batch_size, ind_dataset=ind_dataset, val_ood_dataset=val_dataset, ood_datasets=all_datasets)
    mahalanobis_loaders = get_triplets_loaders(batch_size=20, ind_dataset=ind_dataset, val_ood_dataset=val_dataset, ood_datasets=all_datasets)
    rotation_loaders = get_triplets_loaders(batch_size=1, ind_dataset=ind_dataset, val_ood_dataset=val_dataset, ood_datasets=all_datasets)

    # baseline
    method_loaders = loaders[1:]
    val_ind, val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3 = _baseline(standard_model, method_loaders, device)
    _ood_detection_performance('Baseline', val_ind, val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3, ood_dataset_1, ood_dataset_2, ood_dataset_3)

    # odin
    temp_val_ind, temp_val_ood, temp_ind, temp_ood_1, temp_ood_2, temp_ood_3 = _odin(standard_model, method_loaders, device)
    val_ind, val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3 = _update_scores(val_ind, temp_val_ind, val_ood, temp_val_ood, test_ind, temp_ind, test_ood_1, temp_ood_1, test_ood_2, temp_ood_2, test_ood_3, temp_ood_3, expand=True)
    _ood_detection_performance('Odin', temp_val_ind, temp_val_ood, temp_ind, temp_ood_1, temp_ood_2, temp_ood_3, ood_dataset_1, ood_dataset_2, ood_dataset_3)

    # mahalanobis
    temp_val_ind, temp_val_ood, temp_ind, temp_ood_1, temp_ood_2, temp_ood_3 = _generate_Mahalanobis(standard_model, mahalanobis_loaders, device, num_classes=args.num_classes)
    val_ind, val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3 = _update_scores(val_ind, temp_val_ind, val_ood, temp_val_ood, test_ind, temp_ind, test_ood_1, temp_ood_1, test_ood_2, temp_ood_2, test_ood_3, temp_ood_3)
    _ood_detection_performance('Mahalanobis', temp_val_ind, temp_val_ood, temp_ind, temp_ood_1, temp_ood_2, temp_ood_3, ood_dataset_1, ood_dataset_2, ood_dataset_3)

    # self-supervised
    rotation_loaders = rotation_loaders[1:]
    temp_val_ind, temp_val_ood, temp_ind, temp_ood_1, temp_ood_2, temp_ood_3 = _rotation(rotation_model, rotation_loaders, device, num_classes=args.num_classes)
    val_ind, val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3 = _update_scores(val_ind, temp_val_ind, val_ood, temp_val_ood, test_ind, temp_ind, test_ood_1, temp_ood_1, test_ood_2, temp_ood_2, test_ood_3, temp_ood_3)
    _ood_detection_performance('Self-Supervised', temp_val_ind, temp_val_ood, temp_ind, temp_ood_1, temp_ood_2, temp_ood_3, ood_dataset_1, ood_dataset_2, ood_dataset_3)

    # generalized-odin
    temp_val_ind, temp_val_ood,temp_ind, temp_ood_1, temp_ood_2, temp_ood_3 = _gen_odin_inference(genodin_model, method_loaders, device)
    val_ind, val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3 = _update_scores(val_ind, temp_val_ind, val_ood, temp_val_ood, test_ind, temp_ind, test_ood_1, temp_ood_1, test_ood_2, temp_ood_2, test_ood_3, temp_ood_3)
    _ood_detection_performance('Generalized-Odin', temp_val_ind, temp_val_ood, temp_ind, temp_ood_1, temp_ood_2, temp_ood_3, ood_dataset_1, ood_dataset_2, ood_dataset_3)

    # self-ensemble
    temp_val_ind, temp_val_ood, temp_ind, temp_ood_1, temp_ood_2, temp_ood_3 = _ensemble_inference(ensemble_checkpoints, num_classes, method_loaders, device)
    val_ind, val_ood, test_ind, test_ood_1, test_ood_2, test_ood_3 = _update_scores(val_ind, temp_val_ind, val_ood, temp_val_ood, test_ind, temp_ind, test_ood_1, temp_ood_1, test_ood_2, temp_ood_2, test_ood_3, temp_ood_3)
    _ood_detection_performance('Self-Ensemble', temp_val_ind, temp_val_ood, temp_ind, temp_ood_1, temp_ood_2, temp_ood_3, ood_dataset_1, ood_dataset_2, ood_dataset_3)

    X = np.append(val_ind, val_ood, axis=0)
    y = np.append(np.ones(val_ind.shape[0]), np.zeros(val_ood.shape[0]))

    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    ensemble_ood_lr = LogisticRegression(random_state=global_seed, n_jobs=2, max_iter=200).fit(X, y)
    pred_val_ind = ensemble_ood_lr.predict_proba(val_ind)[:, 1]
    pred_val_ood = ensemble_ood_lr.predict_proba(val_ood)[:, 1]
    _, threshold = _find_threshold(pred_val_ind, pred_val_ood)

    pred_ind = ensemble_ood_lr.predict_proba(test_ind)[:, 1]
    pred_ood_1 = ensemble_ood_lr.predict_proba(test_ood_1)[:, 1]
    auc1, fpr1, acc1 = _score_npzs(pred_ind, pred_ood_1, threshold)

    pred_ood_2 = ensemble_ood_lr.predict_proba(test_ood_2)[:, 1]
    auc2, fpr2, acc2 = _score_npzs(pred_ind, pred_ood_2, threshold)

    pred_ood_3 = ensemble_ood_lr.predict_proba(test_ood_3)[:, 1]
    auc3, fpr3, acc3 = _score_npzs(pred_ind, pred_ood_3, threshold)

    aucs = [auc1, auc2, auc3]
    fprs = [fpr1, fpr2, fpr3]
    accs = [acc1, acc2, acc3]

    method = "Ensemble of OOD detectors"
    _verbose(method, ood_dataset_1, ood_dataset_2, ood_dataset_3, aucs, fprs, accs)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
