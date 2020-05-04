import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from dataLoaders import ISIC19TestSet, PandasDataSetWithPaths
from utils import build_model_with_checkpoint
import argparse
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
import os
import ipdb


# TODO: Check that everything works perfectly here
def _setup_data(csvFilePath='/local_storage/datasets/ISIC2019/ISIC_2019_Test_Metadata.csv', val_batch_size=32, resolution=224):

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor()
    ])

    # testset = ISIC19TestSet(csvFilePath, transform)
    # testset = PandasDataSetWithPaths(csvFilePath, transform)
    # val_indices = np.load('val_indices.npz')['arr_0']
    # val_sampler = SubsetRandomSampler(val_indices)
    #
    # if val_batch_size == -1:
    #     loader = data.DataLoader(testset, batch_size=testset.__len__(), sampler=val_sampler, num_workers=4)
    # else:
    #     loader = data.DataLoader(testset, batch_size=val_batch_size, sampler=val_sampler, num_workers=4)

    testset = ISIC19TestSet(csvFilePath, transform)
    loader = data.DataLoader(testset, batch_size=val_batch_size, num_workers=4)

    return loader


def _predict(model, loader, device, exclude_class=None, state=None, mode='test'):

    paths, results = [], []
    columns = ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    known_classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    if exclude_class is not None:
        idx = known_classes.index(exclude_class)
    model.eval()

    for data in tqdm(loader):
        if mode == 'train':
            path, images, _ = data
        else:
            path, images = data

        images = images.to(device)

        for i in range(len(path)):
            temp = path[i]
            temp = temp.split('/')[-1].split('.jpg')[0]
            paths.append(temp)

        outputs = model(images)
        softmax_outputs = torch.softmax(outputs, 1)
        for output in softmax_outputs:
            temp = output.detach().cpu().numpy().tolist()
            if exclude_class is None:
                results.append([float(elem) for elem in temp]+[1e-30])
            else:
                left_part = temp[:idx]
                right_part = temp[idx:]
                new_elem = [float(x) for x in left_part] + [0.0] + [float(x) for x in right_part] + [1e-30]
                results.append(new_elem)

    df = pd.DataFrame(columns=columns)
    for idx, (path, result) in enumerate(zip(paths, results)):
        df.loc[idx] = [path] + result
    if state is None:
        df.to_csv(f'ISIC2019TestSetpredictionsEx{exclude_class}.csv', index=False)
    else:
        df.to_csv(f'temp_csv_{state}.csv', index=False)
    return f'temp_csv_{state}.csv'


def main(args):

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if args.model_checkpoint is not None:
        model = EfficientNet.from_pretrained('efficientnet-b6')

        exclude_class = args.exclude_class
        if exclude_class is None:
            model._fc = nn.Linear(model._fc.in_features, 8)
        else:
            model._fc = nn.Linear(model._fc.in_features, 7)
        model = model.to(device)

        print('Loading model....')
        state_dict = torch.load(args.model_checkpoint)
        new_state_dict = OrderedDict()

        for key, value in state_dict.items():
            new_key = key.split('module.')[1]
            new_state_dict[new_key] = value

        torch.save(new_state_dict, args.model_checkpoint.split('.pth')[0]+'correct.pth')
        model.load_state_dict(torch.load(args.model_checkpoint.split('.pth')[0]+'correct.pth', map_location=device))
        os.system(f"rm {args.model_checkpoint.split('.pth')[0]+'correct.pth'}")
        print('Done!')
        loader = _setup_data(val_batch_size=int(args.batch_size), resolution=int(args.res))
        loader = _setup_data(csvFilePath="/local_storage/datasets/ISIC2019/Training_paths_and_classes.csv", \
                             val_batch_size=int(args.batch_size), resolution=int(args.res))

        csv_name = _predict(model, loader, device, exclude_class, mode='train')
        df = pd.read_csv(csv_name)
        df = df.set_index('image')
        os.system(f'rm {csv_name}')
        # df = df.div(idx+1)
        # df.to_csv(f'TestSetPredictions.csv')
        df.to_csv(f'TrainSetPredictions.csv')

    elif args.checkpoints_file is not None:

        with open(args.checkpoints_file, 'r') as checkpoints_file:
            for idx, line in tqdm(enumerate(checkpoints_file)):
                model_checkpoint, exclude_class, model_type = line.split('\n')[0].split(', ')
                if exclude_class == 'None':
                    exclude_class = None
                    out_classes = 8
                else:
                    out_classes = 7

                loader = _setup_data(val_batch_size=int(args.batch_size), resolution=int(args.res))
                model = build_model_with_checkpoint(model_type, model_checkpoint, out_classes)
                csv_name = _predict(model, loader, device, exclude_class, state=idx)
                if idx == 0:
                    df = pd.read_csv(csv_name)
                    df = df.set_index('image')
                else:
                    df1 = pd.read_csv(csv_name)
                    df1 = df1.set_index('image')
                    df = df.add(df1)
                os.system(f'rm {csv_name}')
                df.to_csv('TestSetBalanced2.csv')

    # elif args.checkpoints_file is not None:
    #
    #     with open(args.checkpoints_file, 'r') as checkpoints_file:
    #         for idx, line in tqdm(enumerate(checkpoints_file)):
    #             model_checkpoint, exclude_class, model_type = line.split('\n')[0].split(', ')
    #             if exclude_class == 'None':
    #                 exclude_class = None
    #             ipdb.set_trace()
    #             if model_type == 'eb6':
    #                 model = EfficientNet.from_pretrained('efficientnet-b6')
    #                 if exclude_class is None:
    #                     model._fc = nn.Linear(model._fc.in_features, 8)
    #                 else:
    #                     model._fc = nn.Linear(model._fc.in_features, 7)
    #                 model = model.to(device)
    #
    #                 print('Loading model....')
    #                 state_dict = torch.load(os.path.join('./checkpoints', model_checkpoint))
    #                 new_state_dict = OrderedDict()
    #
    #                 for key, value in state_dict.items():
    #                     new_key = key.split('module.')[1]
    #                     new_state_dict[new_key] = value
    #
    #                 torch.save(new_state_dict, os.path.join('./checkpoints', model_checkpoint).split('.pth')[0]+'correct.pth')
    #                 model.load_state_dict(torch.load(os.path.join('./checkpoints', model_checkpoint).split('.pth')[0]+'correct.pth', map_location=device))
    #                 os.system(f"rm {os.path.join('./checkpoints', model_checkpoint).split('.pth')[0]+'correct.pth'}")
    #                 print('Done!')
    #             elif model_type == 'eb0':
    #                 model = EfficientNet.from_pretrained('efficientnet-b0')
    #                 if exclude_class is None:
    #                     model._fc = nn.Linear(model._fc.in_features, 8)
    #                 else:
    #                     model._fc = nn.Linear(model._fc.in_features, 7)
    #                 model = model.to(device)
    #                 model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
    #                 state_dict = torch.load(model_checkpoint, map_location=device)
    #                 model.load_state_dict(state_dict)
    #                 print('Done!')
    #                 print('Loading model....')
    #
    #             loader = _setup_data(val_batch_size=int(args.batch_size), resolution=int(args.res))
    #
    #             csv_name = _predict(model, loader, device, exclude_class, state=idx)
    #             if idx == 0:
    #                 df = pd.read_csv(csv_name)
    #                 df = df.set_index('image')
    #             else:
    #                 df1 = pd.read_csv(csv_name)
    #                 df1 = df1.set_index('image')
    #                 df = df.add(df1)
    #             os.system(f'rm {csv_name}')
    #             # df = df.div(idx+1)
    #             df.to_csv('TestSetBalanced.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test Set Scoring')
    parser.add_argument('--checkpoints_file', '--cf', default=None, required=False)
    parser.add_argument('--model_checkpoint', '--mc', default=None, required=False)
    parser.add_argument('--exclude_class', '--ex', default=None, required=False)
    parser.add_argument('--batch_size', '--bs', type=int, default=10, required=False)
    parser.add_argument('--res', '--r', type=int, default=224, required=False)

    args = parser.parse_args()

    main(args)

