import torch
from torch.autograd import Variable
from models.ResNet import ResNet, BasicBlock
import numpy as np
import argparse
import time
from tqdm import tqdm
from ood import _get_gram_power, _get_layer_deviations
import ipdb


def _gram_matrices(model, loaders, device, num_classes, power=10):

    model.eval()
    train_ind_loader, val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3 = loaders

    temp_x = torch.rand(2, 3, 224, 224).to(device)
    temp_x = Variable(temp_x)
    temp_x = temp_x.to(device)
    x, features = model(temp_x)
    num_feature_maps = len(features)

    mins = [[[None for _ in range(power)] for _ in range(num_feature_maps)] for _ in range(num_classes)]
    maxs = [[[None for _ in range(power)] for _ in range(num_feature_maps)] for _ in range(num_classes)]

    for data in tqdm(train_ind_loader):

        images, _ = data
        images = images.to(device)
        x, features = model(images)
        argmaxs = torch.argmax(x, dim=1)

        for c in range(num_classes):
            indices = np.where(argmaxs.detach().cpu().numpy() == c)
            for layer, feature_map in enumerate(features):
                selected_features = feature_map[indices]
                for p in (range(power)):
                    g_p = _get_gram_power(selected_features, p+1)
                    if g_p is not None:
                        # g_p = g_p.detach().cpu().numpy()
                        channel_mins = g_p.min(dim=0)[0]
                        channel_maxs = g_p.max(dim=0)[0]
                        if mins[c][layer][p] is None:
                            mins[c][layer][p] = channel_mins
                            maxs[c][layer][p] = channel_maxs
                        else:
                            mins[c][layer][p] = torch.min(mins[c][layer][p], channel_mins)
                            maxs[c][layer][p] = torch.min(maxs[c][layer][p], channel_maxs)

    val_ind_deviations = _get_layer_deviations(model, val_ind_loader, device, mins, maxs)
    expectations = np.mean(val_ind_deviations, axis=0)
    val_ind_deviations = np.divide(val_ind_deviations, expectations)
    val_ind_deviations = np.sum(val_ind_deviations, axis=1)

    val_ood_deviations = _get_layer_deviations(model, val_ood_loader, device, mins, maxs)
    val_ood_deviations = np.divide(val_ood_deviations, expectations)
    val_ood_deviations = np.sum(val_ood_deviations, axis=1)

    test_ind_deviations = _get_layer_deviations(model, test_ind_loader, device, mins, maxs)
    test_ind_deviations = np.divide(test_ind_deviations, expectations)
    test_ind_deviations = np.sum(test_ind_deviations, axis=1)

    test_ood_deviations_1 = _get_layer_deviations(model, test_ood_loader_1, device, mins, maxs)
    test_ood_deviations_1 = np.divide(test_ood_deviations_1, expectations)
    test_ood_deviations_1 = np.sum(test_ood_deviations_1, axis=1)

    test_ood_deviations_2 = _get_layer_deviations(model, test_ood_loader_2, device, mins, maxs)
    test_ood_deviations_2 = np.divide(test_ood_deviations_2, expectations)
    test_ood_deviations_2 = np.sum(test_ood_deviations_2, axis=1)

    test_ood_deviations_3 = _get_layer_deviations(model, test_ood_loader_3, device, mins, maxs)
    test_ood_deviations_3 = np.divide(test_ood_deviations_3, expectations)
    test_ood_deviations_3 = np.sum(test_ood_deviations_3, axis=1)

    return val_ind_deviations, val_ood_deviations, test_ind_deviations, test_ood_deviations_1, test_ood_deviations_2, test_ood_deviations_3


if __name__ == '__main__':

    start = time.time()

    parser = argparse.ArgumentParser(description='Out-of-Distribution Detection')

    parser.add_argument('--model_checkpoint', '--mc', required=True)
    parser.add_argument('--in_distribution_dataset', '--in', required=True)
    parser.add_argument('--val_dataset', '--val', required=True)
    parser.add_argument('--test_dataset', '--test', default='tinyimagenet', required=False)
    parser.add_argument('--num_classes', '--nc', type=int, required=True)
    parser.add_argument('--batch_size', '--bs', type=int, default=32, required=False)
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')

    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=10)
    state_dict = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(state_dict, map_location=device)
    model = model.to(device)

