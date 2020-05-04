import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from auto_augment import AutoAugment, Cutout
from Datasets import *
import os
import pickle
import random
import ipdb

abs_path = '/home/ferles/Dermatology/medusa/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _create_gt_csv_file(loader, columns, gtFile, fold_index=None):

    gtFile = os.path.join(abs_path, 'csvs/', gtFile)

    df = pd.DataFrame(columns=columns)
    paths, new_labels = [], []

    for idx, (path, _, labels) in enumerate(loader):

        for i in range(len(path)):
            temp = path[i]
            temp = temp.split('/')[-1].split('.jpg')[0]
            paths.append(temp)

        for label in labels:
            temp = label.detach().cpu().numpy().tolist()
            new_labels.append([float(elem) for elem in temp])

    for idx, (path, result) in enumerate(zip(paths, new_labels)):
        df.loc[idx] = [path] + result

    if fold_index is None:
        df.to_csv(gtFile, index=False)
    else:
        df.to_csv(gtFile.split('.csv')[0]+f'fold{fold_index}'+'.csv', index=False)


def _get_gts(dataset):

    dataloader = DataLoader(dataset, batch_size=100)
    gts = np.zeros(0)
    for (_, _, labels) in tqdm(dataloader):
        _labels = torch.argmax(labels, dim=1)
        gts = np.copy(np.append(gts, _labels.detach().cpu().numpy()))

    return gts


def _get_transforms(input_size, with_auto_augment=False):

    if with_auto_augment:
        training_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.CenterCrop(size=299),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            transforms.RandomRotation(45),
            AutoAugment(),
            Cutout(),
            transforms.ToTensor()
        ])
    else:
        training_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.CenterCrop(size=299),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            transforms.RandomRotation(45),
            transforms.ToTensor()
        ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    return training_transform, val_transform


def _balance_order(data, labels, num_classes, batch_size):

    lists = []
    for _ in range(num_classes):
        lists.append([])

    for idx, elem in enumerate(data):
        class_idx = labels[idx]
        lists[class_idx].append(data[idx])

    final_indexes = []
    subsample_size = int(batch_size / num_classes)

    batch_counter = 0
    num_batches = int(len(data) / batch_size)
    while batch_counter < num_batches:
        acc = []
        for class_idx in range(num_classes):
            cnt = 0
            i = 0
            while (cnt < subsample_size):
                if len(lists[class_idx]) > 0:
                    if lists[class_idx][i] not in acc:
                        acc.append(lists[class_idx][i])
                        del lists[class_idx][i]
                        cnt += 1
                    else:
                        i += 1
        random.shuffle(acc)
        final_indexes.extend(acc)
        batch_counter += 1

    acc = []
    for l in lists:
        acc.extend(l)
    random.shuffle(acc)
    final_indexes.extend(acc)
    return final_indexes


class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.num_samples = len(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples


def oversampling_loaders_custom(csvfiles, train_batch_size, val_batch_size, input_size, gtFile, with_auto_augment=False, load_gts=True, mode='new'):

    training_transform, val_transform = _get_transforms(input_size=input_size, with_auto_augment=with_auto_augment)

    trainset = PandasDataSetWithPaths(csvfiles[0], transform=training_transform)
    testset = PandasDataSetWithPaths(csvfiles[1], transform=val_transform)

    if load_gts:
        gts = np.load(f'{abs_path}npzs/indexes/isic/gts_{mode}.npz')['arr_0']
    else:
        gts = _get_gts(trainset)
        np.savez(f'{abs_path}/npzs/indexes/isic/gts_{mode}.npz', gts)

    indexes = np.array(list(range(gts.shape[0])))
    ros = RandomOverSampler(random_state=global_seed)
    idxs, labels = ros.fit_resample(indexes.reshape(-1, 1), gts)
    idxs = np.squeeze(idxs).tolist()
    labels = np.squeeze(labels).tolist()
    labels = [int(e) for e in labels]
    num_classes = np.unique(labels).shape[0]

    final_idx = _balance_order(idxs, labels, num_classes, batch_size=train_batch_size)
    sampler = SubsetSequentialSampler(indices=final_idx)

    gpu_count = torch.cuda.device_count()
    train_loader = DataLoader(trainset, batch_size=train_batch_size, sampler=sampler, num_workers=16*gpu_count)
    val_loader = DataLoader(testset, batch_size=val_batch_size, num_workers=16*gpu_count)

    _create_gt_csv_file(loader=val_loader, columns=testset.csv_columns, gtFile=gtFile)

    return train_loader, val_loader, testset.csv_columns


def oversampling_loaders_exclude_class_custom_no_gts(csvfiles, train_batch_size, val_batch_size, input_size, gtFile, exclude_class, with_auto_augment=False, load_gts=True, mode='new'):

    training_transform, val_transform = _get_transforms(input_size=input_size, with_auto_augment=with_auto_augment)

    trainset = PandasDataSetWithPaths(csvfiles[0], transform=training_transform, exclude_class=exclude_class)
    valset = PandasDataSetWithPaths(csvfiles[1], transform=val_transform, exclude_class=exclude_class)

    if load_gts:
        gts = np.load(f'{abs_path}npzs/indexes/isic/gts_{exclude_class}_{mode}.npz')['arr_0']
    else:
        gts = _get_gts(trainset)
        np.savez(f'{abs_path}npzs/indexes/isic/gts_{exclude_class}_{mode}.npz', gts)

    indexes = np.array(list(range(gts.shape[0])))
    ros = RandomOverSampler(random_state=global_seed)
    idxs, labels = ros.fit_resample(indexes.reshape(-1, 1), gts)
    idxs = np.squeeze(idxs).tolist()
    labels = np.squeeze(labels).tolist()
    labels = [int(e) for e in labels]
    num_classes = np.unique(labels).shape[0]

    final_idx = _balance_order(idxs, labels, num_classes, batch_size=train_batch_size)
    sampler = SubsetSequentialSampler(indices=final_idx)

    gpu_count = torch.cuda.device_count()
    train_loader = DataLoader(trainset, batch_size=train_batch_size, sampler=sampler, num_workers=16*gpu_count)
    val_loader = DataLoader(valset, batch_size=val_batch_size, num_workers=16*gpu_count)

    _create_gt_csv_file(loader=val_loader, columns=valset.csv_columns, gtFile=gtFile)

    return train_loader, val_loader, valset.csv_columns


def _get_isic_loaders_ood(batch_size,
                          train_csv='/raid/ferles/ISIC2019/folds/Train_Fold_new.csv',
                          val_csv='/raid/ferles/ISIC2019/folds/Val_Fold_new.csv',
                          test_csv='/raid/ferles/ISIC2019/folds/ValFold1.csv',
                          full_csv='/raid/ferles/ISIC2019/Training_paths_and_classes.csv',
                          exclude_class=None):

    input_size = 224
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    trainset = PandasDataSetWithPaths(train_csv, transform=val_transform, exclude_class=exclude_class, ret_path=False)
    valset = PandasDataSetWithPaths(val_csv, transform=val_transform, exclude_class=exclude_class, ret_path=False)
    testset = PandasDataSetWithPaths(test_csv, transform=val_transform, exclude_class=exclude_class, ret_path=False)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=16)
    val_loader = DataLoader(valset, batch_size=batch_size, num_workers=16)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=16)

    if exclude_class is not None:
        ood_set = PandasDataSetSingleClass(full_csv, single_class=exclude_class, transform=val_transform)
        ood_loader = DataLoader(ood_set, batch_size=batch_size, num_workers=16)
        return trainloader, val_loader, testloader, ood_loader
    else:
        return trainloader, val_loader, testloader


def _get_7point_loaders(batch_size, csvfile='/raid/ferles/7-point/7pointAsISIC.csv'):

    input_size = 224
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    val_set = PandasDataSetWithPaths(csvfile, transform=val_transform, exclude_class=None, ret_path=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16)

    return val_loader, val_set.csv_columns


def _get_isic_loader(batch_size, csvfile='/raid/ferles/ISIC2019/Training_paths_and_classes.csv'):

    input_size = 224
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    val_set = PandasDataSetWithPaths(csvfile, transform=val_transform, exclude_class=None, ret_path=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16)

    return val_loader


def _get_7point_loaders_ood(batch_size, exclude_class=None, csvfile='/raid/ferles/7-point/7pointAsISIC.csv', out_mode=False):

    input_size = 224
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    if out_mode == False:
        val_set = PandasDataSetWithPathsExcludes(csvfile, transform=val_transform, exclude_classes=[exclude_class, 'UNK'], ret_path=False)
        ood_dataset = PandasDataSetSingleClass(csvfile, transform=val_transform, single_class=exclude_class)
    else:
        val_set = PandasDataSetWithPaths(csvfile, transform=val_transform, exclude_class=exclude_class, ret_path=False)
        ood_dataset = PandasDataSetSingleClass(csvfile, transform=val_transform, single_class='UNK')

    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, num_workers=16)

    return val_loader, ood_loader


def _get_custom_loader_7point(batch_size, exclude_class, csvfile='/raid/ferles/7-point/7pointdermClasses.csv'):

    input_size = 224
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    ood_dataset = PandasDataSetSingleClass(csvfile, transform=val_transform, single_class=exclude_class)

    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, num_workers=16)

    return ood_loader, ood_loader

def _get_cifar_transforms(resize):

    normalize_cifar = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if resize:

        image_size = 224
        transform_train_cifar = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize_cifar,
        ])

        transform_test_cifar = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize_cifar,
        ])

    else:

        transform_train_cifar = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    return transform_train_cifar, transform_test_cifar


def cifar10loaders(train_batch_size=32, test_batch_size=32, test=False, validation_test_split=0, save_to_pickle=False, pickle_files=None, resize=True):

    transform_train_cifar, transform_test_cifar = _get_cifar_transforms(resize)
    if not test:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_cifar)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test_cifar)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_cifar)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=16)

    if validation_test_split == 0 and pickle_files is None:
        trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=16)
        return trainloader, testloader
    elif pickle_files is not None:
        trainpickle, valpickle = pickle_files
        with open(trainpickle, 'rb') as train_pickle, open(valpickle, 'rb') as val_pickle:
            trainset_indices = pickle.load(train_pickle)
            valset_indices = pickle.load(val_pickle)

            train_sampler = SubsetRandomSampler(trainset_indices)
            test_sampler = SubsetRandomSampler(valset_indices)
            trainloader = DataLoader(trainset, batch_size=train_batch_size, sampler=train_sampler, num_workers=0)
            val_loader = DataLoader(trainset, batch_size=test_batch_size, sampler=test_sampler, num_workers=0)

            return trainloader, val_loader, testloader
    else:

        temp_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=temp_transform_cifar)
        temp_loader = DataLoader(temp_set, batch_size=10000)
        gts = []
        for _, gt in temp_loader:
            gts.extend(gt.detach().cpu().numpy().tolist())
        indexes = list(range(trainset.__len__()))

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_test_split, random_state=global_seed)
        trainset_indices, valset_indices = next(iter(splitter.split(indexes, gts)))

        if save_to_pickle:
            with open('train_indices_cifar10.pickle', 'wb') as train_pickle, open('val_indices_cifar10.pickle', 'wb') as val_pickle:
                pickle.dump(trainset_indices, train_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(valset_indices, val_pickle, protocol=pickle.HIGHEST_PROTOCOL)

        train_sampler = SubsetRandomSampler(trainset_indices)
        test_sampler = SubsetRandomSampler(valset_indices)
        trainloader = DataLoader(trainset, batch_size=test_batch_size, sampler=train_sampler, num_workers=16)
        val_loader = DataLoader(trainset, batch_size=test_batch_size, sampler=test_sampler, num_workers=16)

        return trainloader, val_loader, testloader


def create_ensemble_loaders(train_batch_size=32, test_batch_size=32, k=5, num_classes=10, pickle_files=None, resize=True):

    transform_train_cifar, transform_test_cifar = _get_cifar_transforms(resize)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_cifar)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_cifar)

    if pickle_files is not None:
        trainpickle, _ = pickle_files
        with open(trainpickle, 'rb') as train_pickle:
            trainset_indices = pickle.load(train_pickle)
            trainset_indices = list(trainset_indices)

    unique_labels = list(np.random.permutation(num_classes))
    step = len(unique_labels) // k
    point = 0

    train_ind_loaders, train_ood_loaders = [], []
    test_ind_loaders, test_ood_loaders = [], []

    while point < len(unique_labels):

        temp_labels = unique_labels[point: min(len(unique_labels), point+step)]

        custom_trainset_ind = CustomEnsembleDatasetIn(trainset, remove_labels=temp_labels, keep_indices=trainset_indices)
        custom_trainset_out = CustomEnsembleDatasetOut(trainset, remove_labels=temp_labels, keep_indices=trainset_indices)

        train_ind_sampler = SubsetRandomSampler(custom_trainset_ind.keep_indices)
        train_ood_sampler = SubsetRandomSampler(custom_trainset_out.keep_indices)

        train_ind_loader = DataLoader(custom_trainset_ind, batch_size=train_batch_size, sampler=train_ind_sampler)
        train_ood_loader = DataLoader(custom_trainset_out, batch_size=train_batch_size, sampler=train_ood_sampler)

        train_ind_loaders.append(train_ind_loader)
        train_ood_loaders.append(train_ood_loader)

        testset_indices = list(range(testset.__len__()))
        custom_testset_ind = CustomEnsembleDatasetIn(testset, remove_labels=temp_labels, keep_indices=testset_indices)
        custom_testset_out = CustomEnsembleDatasetOut(testset, remove_labels=temp_labels, keep_indices=testset_indices)

        test_ind_sampler = SubsetRandomSampler(custom_testset_ind.keep_indices)
        test_ood_sampler = SubsetRandomSampler(custom_testset_out.keep_indices)

        test_ind_loader = DataLoader(custom_testset_ind, batch_size=test_batch_size, sampler=test_ind_sampler)
        test_ood_loader = DataLoader(custom_testset_out, batch_size=test_batch_size, sampler=test_ood_sampler)

        test_ind_loaders.append(test_ind_loader)
        test_ood_loaders.append(test_ood_loader)

        point += step

    return train_ind_loaders, train_ood_loaders, test_ind_loaders, test_ood_loaders


def cifar100loaders(train_batch_size=32, test_batch_size=32, test=False, validation_test_split=0, save_to_pickle=False, pickle_files=None, resize=True):

    transform_train_cifar, transform_test_cifar = _get_cifar_transforms(resize=resize)
    if not test:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_cifar)
    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test_cifar)
    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test_cifar)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=16)

    if validation_test_split == 0:
        return trainloader, testloader
    else:

        if pickle_files is None:
            temp_test = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=temp_transform_cifar)
            temp_loader = DataLoader(temp_test, batch_size=1000)
            gts = []
            for _, gt in temp_loader:
                gts.append(gt.detach().cpu().numpy().tolist())
            indexes = list(range(trainset.__len__()))

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_test_split, random_state=global_seed)
            trainset_indices, valset_indices = next(iter(splitter.split(indexes, gts)))

            if save_to_pickle:
                with open('train_indices_cifar100.pickle', 'wb') as train_pickle, open('val_indices_cifar100.pickle', 'wb') as val_pickle:
                    pickle.dump(trainset_indices, train_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(valset_indices, val_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            trainpickle, valpickle = pickle_files
            with open(trainpickle, 'rb') as train_pickle, open(valpickle, 'rb') as val_pickle:
                trainset_indices = pickle.load(train_pickle)
                valset_indices = pickle.load(val_pickle)

        train_sampler = SubsetRandomSampler(trainset_indices)
        test_sampler = SubsetRandomSampler(valset_indices)
        trainloader = DataLoader(trainset, batch_size=test_batch_size, sampler=train_sampler, num_workers=16)
        val_loader = DataLoader(trainset, batch_size=test_batch_size, sampler=test_sampler, num_workers=16)

        return trainloader, val_loader, testloader


def tinyImageNetloader(batch_size, resize=True):

    normalize_cifar = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if resize:
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize_cifar,
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize_cifar,
        ])

    dataset = datasets.ImageFolder(root='/raid/ferles/tiny-imagenet-200/test', transform=transform_test)
    loader = DataLoader(dataset, batch_size=batch_size)

    return loader