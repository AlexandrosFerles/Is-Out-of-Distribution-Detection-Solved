import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.utils import data
from torchvision import transforms
from dataLoaders import ISIC19TestSet
import argparse
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
import os
import random
import ipdb

global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _predict(model, loader, device):

    paths, results = [], []
    columns = ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']

    for data in tqdm(loader):
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
            results.append([float(elem) for elem in temp])

    df = pd.DataFrame(columns=columns)
    for idx, (path, result) in enumerate(zip(paths, results)):
        df.loc[idx] = [path] + result
    df.to_csv(f'ISIC2019predictionsExtra.csv', index=False)


def _setup_data(csvFilePath='/local_storage/datasets/ISIC2019/ISIC_2019_Test_Metadata.csv', val_batch_size=4, resolution=224):

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor()
    ])

    testset = ISIC19TestSet(csvFilePath, transform)
    loader = data.DataLoader(testset, batch_size=val_batch_size, num_workers=4)

    return loader


def main(args):
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if args.model.lower() == 'efficientnet6':
        model = EfficientNet.from_pretrained('efficientnet-b6')                    
        model._fc = nn.Linear(model._fc.in_features, 9)
        model = model.to(device)
    else:
        raise NotImplementedError

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

    _predict(model, loader, device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test Set Scoring')
    parser.add_argument('--model', '--m', required=True)
    parser.add_argument('--model_checkpoint', '--mc', required=True)
    parser.add_argument('--batch_size', '--bs', type=int, default=4, required=False)
    parser.add_argument('--res', '--r', type=int, default=224, required=False)

    args = parser.parse_args()

    main(args)
