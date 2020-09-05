from  __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform
from tqdm import tqdm
import random
import ipdb

device = torch.device('cuda:1')
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)



# this function is from https://github.com/xingjunm/lid_adversarial_subspace_detection
def merge_and_generate_labels(X_pos, X_neg):
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y


def sample_estimator(model, num_classes, feature_list, train_loader, device, features_mode='all'):

    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for datas in tqdm(train_loader):

        data, target = datas

        total += data.size(0)
        data = data.to(device)
        with torch.no_grad():
            data = Variable(data)
            data = data.to(device)
            if features_mode == 'all':
                output, out_features = model.extract_features(data, mode=features_mode)
                idxs = [0, 2, 4, 7, 10, 14, 15]
                out_features = [out_features[idx] for idx in idxs] + [output]
            else:
                output = model.extract_features(data, mode=features_mode)
                out_features = [output]

            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)

            for i in range(data.size(0)):
                if len(target[i].size()) == 0:
                    label = target[i]
                else:
                    label = torch.argmax(target[i]).item()

                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                            = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1
                num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
        for j_ in range(num_classes):
            temp_list[j_] = torch.mean(list_features[out_count][j_], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().to(device)
        precision.append(temp_precision)

    return sample_class_mean, precision


def get_Mahalanobis_score(model, test_loader, num_classes, sample_mean, precision, layer_index, magnitude, device, features_mode='all'):

    model.eval()
    model = model.to(device)
    Mahalanobis = []

    for input in tqdm(test_loader):

        data, _ = input

        data = data.to(device)
        data = Variable(data, requires_grad=True)

        if features_mode == 'all':
            idxs = [0, 2, 4, 7, 10, 14, 15]
            x, features = model.extract_features(data, mode=features_mode)
            features = [features[idx].to(device) for idx in idxs] + [x.to(device)]
        else:
            x = model.extract_features(data, mode=features_mode)
            features = [x]

        out_features = features[layer_index]
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i].to(device)
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f.to(device), precision[layer_index].to(device)), zero_f.t()).to(device).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred).to(device)
        zero_f = out_features - Variable(batch_sample_mean).to(device)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index]).to(device)), zero_f.t()).to(device).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        tempInputs = torch.add(data.data, -magnitude, gradient)

        if features_mode == 'all':
            x_noise, noise_out_features = model.extract_features(Variable(tempInputs), mode=features_mode)
            noise_out_features = [noise_out_features[idx].to(device) for idx in idxs] + [x_noise.to(device)]
            noise_out_features = noise_out_features[layer_index]
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
        else:
            x_noise = model.extract_features(Variable(tempInputs), mode=features_mode)
            noise_out_features = [x_noise]
            noise_out_features = noise_out_features[layer_index]
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)

        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i].to(device)
            zero_f = noise_out_features.to(device).data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f.to(device), precision[layer_index].to(device)), zero_f.t()).to(device).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1).to(device)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1).to(device)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())

    return Mahalanobis
