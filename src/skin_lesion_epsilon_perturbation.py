import torch
import numpy as np
from torch.autograd import Variable
from utils import build_model_with_checkpoint
from dataLoaders import get_ood_loaders
from tqdm import tqdm
import lib_generation
import argparse
import os
import random
from ood import _get_odin_scores, _score_mahalanobis, _find_threshold, _score_npzs
from ood_ensemble import _ood_detection_performance
import ipdb

abs_path = '/home/ferles/Dermatology/medusa/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _ood_detection_performance(method, val_ind, val_ood, test_ind, test_ood):

    _, threshold = _find_threshold(val_ind, val_ood)

    auc, fpr, acc = _score_npzs(test_ind, test_ood, threshold)

    print()
    print(f'Method: {method}')
    print(f'Area Under Receiver Operating Characteristic curve: {auc}')
    print(f'True Negative Rate @ 95% True Positive Rate: {100-fpr}')
    print(f'Detection Accuracy: {acc}')


def _odin(model, loaders, device):

    model.eval()
    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader = loaders


    for T in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        print(f'########## T: {T} ##########')
        print()
        for epsilon in tqdm([-0.1, -0.0005, 0, 0.0005, 0.1]):

            val_ind = _get_odin_scores(model, val_ind_loader, T, epsilon, device=device)
            val_ood = _get_odin_scores(model, val_ood_loader, T, epsilon, device=device)

            ind = _get_odin_scores(model, test_ind_loader, T, epsilon, device=device)
            ood = _get_odin_scores(model, test_ood_loader, T, epsilon, device=device)

            print(f'########## epsilon: {epsilon} ##########')
            _ood_detection_performance('Odin', val_ind, val_ood, ind, ood)
        print()


def _generate_Mahalanobis(model, loaders, device, num_classes, model_type='eb0'):

    model.eval()
    train_ind_loader, val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader = loaders

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
            M_ood = lib_generation.get_Mahalanobis_score(model, test_ood_loader, num_classes, sample_mean, precision, i, magnitude, device=device)
            M_ood = np.asarray(M_ood, dtype=np.float32)
            if i == 0:
                Mahalanobis_ood = M_ood.reshape((M_ood.shape[0], -1))
            else:
                Mahalanobis_ood = np.concatenate((Mahalanobis_ood, M_ood.reshape((M_ood.shape[0], -1))), axis=1)

        Mahalanobis_test = np.asarray(Mahalanobis_test, dtype=np.float32)
        Mahalanobis_ood = np.asarray(Mahalanobis_ood, dtype=np.float32)

        ind = regressor.predict_proba(Mahalanobis_test)[:, 1]
        ood = regressor.predict_proba(Mahalanobis_ood)[:, 1]

        print(f'########## epsilon: {magnitude} ##########')
        _ood_detection_performance('Mahalanobis', val_ind, val_ood, ind, ood)


if __name__ == '__main__':

    import time
    start = time.time()

    parser = argparse.ArgumentParser(description='Out-of-Distribution Detection')

    parser.add_argument('--model_checkpoint', '--mc', default=None, required=True)
    parser.add_argument('--mode', '--md', type=int, default=0, required=False)
    parser.add_argument('--num_classes', '--nc', type=int, required=True)
    parser.add_argument('--batch_size', '--bs', type=int, default=32, required=False)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')

    standard_checkpoint = args.model_checkpoint
    standard_model = build_model_with_checkpoint('eb0', standard_checkpoint, device=device, out_classes=args.num_classes)

    if args.mode == 0:
        loaders = get_ood_loaders('isic', 'imagenet', 'places', batch_size=args.batch_size)
        mahalanobis_loaders = get_ood_loaders('isic', 'imagenet', 'places', batch_size=20)
    elif args.mode == 1:
        loaders = get_ood_loaders('isic', 'imagenet', 'dermofit-in', batch_size=args.batch_size)
        mahalanobis_loaders = get_ood_loaders('isic', 'imagenet', 'dermofit-in', batch_size=20)
    elif args.mode == 2:
        loaders = get_ood_loaders('isic', 'imagenet', 'dermofit-out', batch_size=args.batch_size)
        mahalanobis_loaders = get_ood_loaders('isic', 'imagenet', 'dermofit-out', batch_size=20)

    _odin(standard_model, loaders[1:], device)
    # _generate_Mahalanobis(standard_model, mahalanobis_loaders, device, num_classes=args.num_classes)