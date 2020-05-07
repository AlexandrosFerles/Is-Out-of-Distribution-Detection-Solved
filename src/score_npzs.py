import numpy as np
from sklearn.metrics import roc_curve, auc
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Npzs Eval')
    parser.add_argument('--known', '--kn', required=True)
    parser.add_argument('--novel', '--nv', required=True)

    args = parser.parse_args()

    ind = np.load(args.known)['arr_0']
    ood = np.load(args.novel)['arr_0']

    y_known, y_novel = np.ones(ind.shape[0]), np.zeros(ood.shape[0])
    X, y = np.append(ind, ood), np.append(y_known, y_novel)

    fpr, tpr, _ = roc_curve(y, X)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(100*roc_auc, 2)

    perc = 0.95
    sorted_ = np.sort(ind)
    threshold = sorted_[int((1-perc)*ind.shape[0])]

    ind_ = np.zeros(ind.shape)
    ood_ = np.zeros(ood.shape)
    ind_[np.argwhere(ind > threshold)] = 1
    ood_[np.argwhere(ood > threshold)] = 1

    X = np.append(ind_, ood_)
    bool_X = np.atleast_1d(X.astype(np.bool))
    bool_y = np.atleast_1d(y.astype(np.bool))

    tn = np.count_nonzero(~bool_X & ~bool_y)
    fp = np.count_nonzero(bool_X & ~bool_y)

    fpr = round(100*fp/(fp+tn), 2)

    print(f"AUC: {roc_auc}")
    print(f"FPR: {fpr}")
