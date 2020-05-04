import numpy as np
from matplotlib import pyplot as plt
import argparse

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.utils import shuffle

import pickle
import ipdb


def _compute_regions(X, Y, auroc, auprin, auprout):

    # TODO: Update it for this case
    fpr, tpr, thresholds = roc_curve(Y, X)

    plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
    plt.plot(fpr, tpr, marker='.', label=f'Mahalanobis, AUC: {auroc}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    # plt.title(f"ROC curve CIFAR vs SVHN")
    # plt.savefig(f'roc_curve_cifar_in_svhn_out.png')

    plt.clf()
    no_skill = Y[Y==1.0].shape[0] / Y.shape[0]
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
    precision, recall, _ = precision_recall_curve(Y, X)
    plt.plot(recall, precision, label=f'AUPRIN: {auprin}\nAUPROUT: {auprout}\n')
    plt.legend(loc="lower left")
    plt.xlabel("recall")
    plt.ylabel("precision")
    # plt.title(f"Precision vs. Recall curve CIFAR vs SVHN")
    # plt.savefig(f'pr_curve_cifar_in_svhn_out.png')


def _get_curve(known, novel):

    known.sort()
    novel.sort()
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]
        tpr95_pos = np.abs(tp / num_k - .95).argmin()
        tnr_at_tpr95 = 1. - fp[tpr95_pos] / num_n

    return tp, fp, tnr_at_tpr95


def _metric(tp, fp, tnr_at_tpr95, verbose=True):

    mtypes = ['TNR@TPR95%', 'AUROC', 'DTACC', 'AUPR_IN', 'AUPR_OUT']

    # TNR
    mtype = mtypes[0]
    if verbose:
        print(f'{mtype}: {tnr_at_tpr95}')

    # AUROC
    mtype = mtypes[1]
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    auroc = -np.trapz(1.-fpr, tpr)
    if verbose:
        print(f'{mtype}: {auroc}')

    # DTACC
    mtype = mtypes[2]
    dtacc = .5 * (tp/tp[0] + 1.-fp/fp[0]).max()
    if verbose:
        print(f'{mtype}: {dtacc}')

    # AUIN
    mtype = mtypes[3]
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    auprin = -np.trapz(pin[pin_ind], tpr[pin_ind])
    if verbose:
        print(f'{mtype}: {auprin}')

    # AUOUT
    mtype = mtypes[4]
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    auprout = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
    if verbose:
        print(f'{mtype}: {auprout}')

    return auroc, dtacc, auprin, auprout


def main(args):

    known = np.load(args.known)['arr_0']
    novel = np.load(args.novel)['arr_0']

    tp, fp, tnr_at_tpr95 = _get_curve(known, novel)
    auroc, dtacc, auprin, auprout = _metric(tp, fp, tnr_at_tpr95)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Npzs Eval')
    parser.add_argument('--known', '--k', required=True)
    parser.add_argument('--novel', '--nv', required=True)
    parser.add_argument('--exclude_class', '--ex', required=True)

    args = parser.parse_args()

    main(args)
