import torch
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
from utils import build_model_with_checkpoint
from dataLoaders import get_ood_loaders
from tqdm import tqdm
import lib_generation
import argparse
import os
import random
from ood import _get_metrics, _score_npzs, _find_threshold, _get_odin_scores
from ood_ensemble import _ood_detection_performance
import ipdb

abs_path = '/home/ferles/Dermatology/medusa/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _odin(model, loaders, device, ind_dataset, ood_dataset, ood_dataset_1, ood_dataset_2, ood_dataset_3):

    model.eval()
    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders

    T = 1000

    for epsilon in tqdm(-0.1, -0.0005, 0, 0.0005, 0.1):

        val_ind = _get_odin_scores(model, val_ind_loader, T, epsilon, device=device)
        val_ood = _get_odin_scores(model, val_ood_loader, T, epsilon, device=device)

        ind = _get_odin_scores(model, test_ind_loader, T, epsilon, device=device)
        ood_1 = _get_odin_scores(model, test_ood_loader_1, T, epsilon, device=device)
        ood_2 = _get_odin_scores(model, test_ood_loader_2, T, epsilon, device=device)
        ood_3 = _get_odin_scores(model, test_ood_loader_3, T, epsilon, device=device)

        print(f'########## epsilon: {epsilon} ##########')
        _ood_detection_performance('Odin', val_ind, val_ood, ind, ood_1, ood_2, ood_3, ood_dataset_1, ood_dataset_2, ood_dataset_3)



def _generate_Mahalanobis(model, loaders, device, ind_dataset, ood_dataset, num_classes, model_type='eb0'):

    model.eval()
    train_ind_loader, val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader = loaders

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

    m_list = [-0.01, -0.0005, 0.0, 0.0005, 0.01]
    for magnitude in m_list:

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
        Mahalanobis_ood = np.asarray(Mahalanobis_test, dtype=np.float32)

        regressor, auc, threshold = _score_mahalanobis(Mahalanobis_test, Mahalanobis_test)
        auc, fpr, acc = _predict_mahalanobis(regressor, Mahalanobis_test, Mahalanobis_ood, threshold)

        print(f'Mahalanobis results on {ind_dataset} (In) vs {ood_dataset} (Out)  with epsilon: {magnitude}:')
        print(f'Area Under Receiver Operating Characteristic curve: {auc}')
        print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')
        print(f'Detection Accuracy : {acc}')
        print('###############################################')
        print()


if __name__ == '__main__':

    import time
    start = time.time()

    parser = argparse.ArgumentParser(description='Out-of-Distribution Detection')

    parser.add_argument('--ood_method', '--m', required=True)
    parser.add_argument('--num_classes', '--nc', type=int, required=True)
    parser.add_argument('--in_distribution_dataset', '--in', required=True)
    parser.add_argument('--out_distribution_dataset', '--out', required=True)
    parser.add_argument('--model_checkpoint', '--mc', default=None, required=False)
    parser.add_argument('--batch_size', '--bs', type=int, default=32, required=False)
    parser.add_argument('--subset_index', '--sub', default=None, required=False)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')

    ood_method = args.ood_method.lower()

    if args.model_checkpoint is None and args.model_checkpoints_file is None:
        raise NotImplementedError('You need to specify either a single or multiple checkpoints')
    elif args.model_checkpoint is not None:
        if ood_method == 'self-supervision' or ood_method == 'selfsupervision' or ood_method =='self_supervision' or ood_method =='rotation':
            model = build_model_with_checkpoint('roteb0', args.model_checkpoint, device=device, out_classes=args.num_classes)
        elif ood_method == 'generalized-odin' or ood_method == 'generalizedodin':
            model = build_model_with_checkpoint('geneb0', args.model_checkpoint, device=device, out_classes=args.num_classes)
        else:
            model = build_model_with_checkpoint('eb0', args.model_checkpoint, device=device, out_classes=args.num_classes)
    else:
        model_checkpoints, num_classes = [], []
        for line in open(args.model_checkpoints_file, 'r'):
            model_checkpoint, nc = line.split('\n')[0].split(',')
            nc = int(nc)
            model_checkpoints.append(model_checkpoint)
            num_classes.append(nc)

    loaders = get_ood_loaders(batch_size=args.batch_size, ind_dataset=args.in_distribution_dataset, val_ood_dataset='cifar10', test_ood_dataset=args.out_distribution_dataset, exclude_class=None, subset_index=args.subset_index)

    if ood_method == 'odin':
        method_loaders = loaders[1:]
        _odin(model, method_loaders, device, ind_dataset=args.in_distribution_dataset, ood_dataset=args.out_distribution_dataset)
    elif ood_method == 'mahalanobis':
        _generate_Mahalanobis(model, loaders=loaders, ind_dataset=args.in_distribution_dataset, ood_dataset=args.out_distribution_dataset, num_classes=args.num_classes, device=device)
    else:
        raise NotImplementedError('Requested unknown Out-of-Distribution Detection Method')

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
