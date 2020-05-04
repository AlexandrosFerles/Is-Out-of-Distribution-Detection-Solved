import wandb
import pandas as pd
import numpy as np
import ipdb


def wandb_logging(evaluation_metrics):

    aucs, aucs_sens, avg_precs, accuracies, sensitivities, specificities, dices, ppvs, npvs = evaluation_metrics

    wandb.log({f"AUC": aucs})
    wandb.log({f"AUC>80%": aucs_sens})
    wandb.log({f"Average precision%": avg_precs})
    wandb.log({f"Accuracy": accuracies})
    wandb.log({f"Sensitivity": sensitivities})
    wandb.log({f"Specificity": specificities})
    wandb.log({f"F1-score": dices})
    wandb.log({f"PPV": ppvs})
    wandb.log({f"NPV": npvs})


def wandb_table(txt_file, epoch=0, num_classes=8, fold_index=None):

    metrics, data, auc, balanced_accuracy = _log_output(txt_file, num_classes)

    # ipdb.set_trace()
    if fold_index is None:
        wandb.log({f"Evaluation Metrics": wandb.Table(data=data, columns=metrics), 'epoch': epoch})
    else:
        wandb.log({f"Evaluation Metrics - Fold {fold_index}": wandb.Table(data=data, columns=metrics), 'epoch': epoch})

    return auc, balanced_accuracy


def _log_output(txt_file, num_classes=8):

    # ipdb.set_trace()
    lines = []
    temp = open(txt_file, 'r')
    for line in temp:
        lines.append(line)

    lines = lines[4:]
    lines = [line for line in lines if line !='\n']
    lines = [line.split('\n')[0] for line in lines]
    metrics = [_capitalize_first_letter(elem) for elem in lines[0].split(' ') if len(elem) > 0]
    metrics = ['Metrics'] + metrics
    metrics[-1] = 'Average Precision'

    idx = metrics.index('Auc')

    data = []
    individual_class_metrics = [line.split(' ') for line in lines[1:1+num_classes]]
    for class_metric in individual_class_metrics:

        temp = [elem for elem in class_metric if len(elem) > 0]
        tag = temp[0]
        results = [float(elem) for elem in temp[1:]]
        data.append([tag] + results)

    mean_values_line = []
    mean_values = lines[2+num_classes:11+num_classes]
    for mean_value in mean_values:
        mean_value = mean_value.split(' ')
        mean_values_line.append(float(mean_value[-1]))

    auc = mean_values_line[idx-1]
    data.append(['Mean Value'] + mean_values_line)

    balanced_accuracy = float(lines[-1].split(' ')[-1])

    return metrics, data, auc, balanced_accuracy


def _capitalize_first_letter(s):

    return s[0].upper()+s[1:]




