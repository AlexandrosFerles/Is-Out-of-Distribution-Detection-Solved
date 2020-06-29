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
import ipdb

abs_path = '/home/ferles/Dermatology/medusa/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _find_threshold(train_scores, val_scores):

    probs = np.append(train_scores, val_scores, axis=0)
    y_true = np.append(np.ones(train_scores.shape[0]), np.zeros(val_scores.shape[0]), axis=0)
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(balanced_accuracy_score(y_true, [1 if m > thresh else 0 for m in probs]))

    accuracies = np.array(accuracy_scores)
    max_accuracy = round(100*accuracies.max(), 2)
    max_accuracy_threshold = thresholds[accuracies.argmax()]

    print(f'Chosen threshold: {max_accuracy_threshold} yielding {max_accuracy}% accuracy')
    return max_accuracy, max_accuracy_threshold


def _score_classification_accuracy(model, testloader, device, dataset, genOdin=False):

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
            if 'isic' in dataset:
                _labels = torch.argmax(labels, dim=1)
                softmax_outputs = torch.softmax(outputs, 1)
                max_idx = torch.argmax(softmax_outputs, axis=1)
                total += max_idx.size()[0]
                correct += (max_idx == _labels).sum().item()
            else:
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = round(100*correct/total, 2)
    print(f'Accuracy on {dataset}: {accuracy}%')


def _get_metrics(X, y):

    tn, fp, fn, tp = confusion_matrix(y, X).ravel()
    fpr = fp/(fp+tn)
    acc = balanced_accuracy_score(y, X)

    return fpr, acc


def _score_npzs(ind, ood, threshold):

    y_known, y_novel = np.ones(ind.shape[0]), np.zeros(ood.shape[0])
    X, y = np.append(ind, ood, axis=0), np.append(y_known, y_novel, axis=0)

    fpr, tpr, _ = roc_curve(y, X)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(100*roc_auc, 2)

    perc = 0.95
    sorted_ = np.sort(ind)
    fpr_threshold = sorted_[int((1-perc)*ind.shape[0])]

    ind_ = np.zeros(ind.shape)
    ood_ = np.zeros(ood.shape)
    ind_[np.argwhere(ind >= fpr_threshold)] = 1
    ood_[np.argwhere(ood >= fpr_threshold)] = 1
    X_fpr = np.append(ind_, ood_, axis=0)

    fpr_, _ = _get_metrics(X_fpr, y)
    fpr_ = round(100*fpr_, 2)

    ind_acc = np.zeros(ind.shape)
    ood_acc = np.zeros(ood.shape)
    ind_acc[np.argwhere(ind >= threshold)] = 1
    ood_acc[np.argwhere(ood >= threshold)] = 1
    X_acc = np.append(ind_acc, ood_acc, axis=0)

    _, acc = _get_metrics(X_acc, y)
    acc = round(100*acc, 2)

    return roc_auc, fpr_, acc


def _score_mahalanobis(ind, ood):

    y_known = np.ones(ind.shape[0])
    y_novel = np.zeros(ood.shape[0])

    X = np.append(ind, ood, axis=0)
    y = np.append(y_known, y_novel)

    lr = LogisticRegressionCV(n_jobs=-1, cv=5, max_iter=1000).fit(X, y)

    known_preds = lr.predict_proba(ind)[:, 1]
    novel_preds = lr.predict_proba(ood)[:, 1]

    _, threshold = _find_threshold(known_preds, novel_preds)

    fpr, tpr, _ = roc_curve(y, np.append(known_preds, novel_preds, axis=0))
    roc_auc = round(100*auc(fpr, tpr), 2)

    return lr, roc_auc, threshold


def _predict_mahalanobis(regressor, ind, ood, threshold):

    known_preds = regressor.predict_proba(ind)[:, 1]
    novel_preds = regressor.predict_proba(ood)[:, 1]
    auc, fpr, acc = _score_npzs(known_preds, novel_preds, threshold)

    return auc, fpr, acc


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


def _odin(model, loaders, device, ind_dataset, ood_dataset):

    model.eval()
    val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader = loaders

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

    _, threshold = _find_threshold(best_val_ind, best_val_ood)

    ind = _get_odin_scores(model, test_ind_loader, best_T, best_epsilon, device=device)
    ood = _get_odin_scores(model, test_ood_loader, best_T, best_epsilon, device=device)

    ind_savefile_name = f'npzs/odin_{ind_dataset}_ind_{ind_dataset}_ood_{ood_dataset}_temperature_{best_T}_epsilon{best_epsilon}_{exclude_class}.npz'
    ood_savefile_name = f'npzs/odin_{ood_dataset}_ind_{ind_dataset}_ood_{ood_dataset}_temperature_{best_T}_epsilon{best_epsilon}_{exclude_class}.npz'

    np.savez(ind_savefile_name, ind)
    np.savez(ood_savefile_name, ood)

    auc, fpr, acc = _score_npzs(ind, ood, threshold)

    print('###############################################')
    print()
    print(f'Succesfully stored in-distribution ood scores to {ind_savefile_name} and out-distribution ood scores to: {ood_savefile_name}')
    print()
    print('###############################################')
    print()
    print(f"Odin results on {ind_dataset} (In) vs {ood_dataset} (Out) and chosen T={best_T}, epsilon={best_epsilon}:")
    print(f'Area Under Receiver Operating Characteristic curve: {auc}')
    print(f'False Positive Rate @ 95% True Positive Rate: {fpr}')
    print(f'Detection Accuracy: {acc}')


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
                pass
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
