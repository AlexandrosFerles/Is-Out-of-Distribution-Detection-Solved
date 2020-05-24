import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms, datasets
from torchvision.transforms import functional as Ftransforms
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
from auto_augment import AutoAugment, Cutout
from Datasets import *
import os
import pickle
import random
import ipdb

abs_path = '/home/ferles/medusa/src/'
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

    dataloader = DataLoader(dataset, batch_size=1000)
    gts = np.zeros(0)
    for ( _, labels) in tqdm(dataloader):
        _labels = torch.argmax(labels, dim=1)
        gts = np.copy(np.append(gts, _labels.detach().cpu().numpy()))

    return gts


def _get_transforms():

    normalize = transforms.Normalize((0.6796, 0.5284, 0.5193), (0.1200, 0.1413, 0.1538))

    training_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
        AutoAugment(),
        transforms.RandomCrop(224),
        Cutout(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    return training_transform, val_transform


def _get_crop_transforms(input_size, with_auto_augment=False, random_crop=True):

    if with_auto_augment:
        if random_crop:
            training_transform = transforms.Compose([
                transforms.RandomCrop(input_size),
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
                transforms.CenterCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
                transforms.RandomRotation(45),
                AutoAugment(),
                Cutout(),
                transforms.ToTensor()
            ])
    else:
        if random_crop:
            training_transform = transforms.Compose([
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
                transforms.RandomRotation(45),
                transforms.ToTensor()
            ])
        else:
            training_transform = transforms.Compose([
                transforms.CenterCrop(input_size),
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


def _get_image_size(img):
    if Ftransforms._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class OrderedCrops(object):
    """
    Ordered crops of an image
    """
    def __init__(self, crop_size, ncrops, pad_if_needed=True):
        self.pad_if_needed = pad_if_needed
        self.crop_size = crop_size
        self.ncrops = ncrops

    @staticmethod
    def get_params(img, crop_size, ncrops):
        width, height = _get_image_size(img)
        if width < crop_size:
            height = int(crop_size/float(width)) * height
            width = crop_size
        if height < crop_size:
            width = int(crop_size/float(height))*width
            height = crop_size
        crop_positions = np.zeros((ncrops, 2))
        ind = 0
        for i in range(np.int64(np.sqrt(ncrops))):
            for j in range(np.int64(np.sqrt(ncrops))):
                crop_positions[ind, 0] = np.int64(crop_size/2 + i*(width-crop_size)/(np.int64(np.sqrt(ncrops))-1))
                crop_positions[ind, 1] = np.int64(crop_size/2 + j*(height-crop_size)/(np.int64(np.sqrt(ncrops))-1))
                ind += 1

        return crop_positions

    def __call__(self, img):

        crop_positions = self.get_params(img, crop_size=self.crop_size, ncrops=self.ncrops)
        width, height = _get_image_size(img)

        if self.pad_if_needed and img.size[0] < self.crop_size:
            img = F.pad(img, (self.crop_size-img.size[0], 0))
        if self.pad_if_needed and img.size[1] < self.crop_size:
            img = F.pad(img, (0, self.crop_size-img.size[1]))

        cropped_image = img.crop((
                        crop_positions[0, 0]-self.crop_size,
                        crop_positions[0, 1]-self.crop_size,
                        (crop_positions[0, 0]-self.crop_size)+self.crop_size,
                        (crop_positions[0, 1]-self.crop_size) + self.crop_size
                        ))

        temp = [cropped_image]

        for i in range(1, self.ncrops):

            cropped_image = img.crop((
                crop_positions[i, 0]-self.crop_size,
                crop_positions[i, 1]-self.crop_size,
                (crop_positions[i, 0]-self.crop_size)+self.crop_size,
                (crop_positions[i, 1]-self.crop_size) + self.crop_size
                ))
            temp.append(cropped_image)

        return tuple(temp)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def generate_random_multi_crop_loader(csvfiles, ncrops, train_batch_size, gtFile, with_auto_augment=False, input_size=224, oversample=False):

    center_transform, _ = _get_crop_transforms(input_size=input_size, with_auto_augment=with_auto_augment, random_crop=False)
    temp_trainset = PandasDataSetWithPaths(csvfiles[0], transform=center_transform, ret_path=False)
    datasets = [temp_trainset]
    n_train_crops, n_val_crops = ncrops
    for i in range(n_train_crops-1):
        random_transform, _ = _get_crop_transforms(input_size=input_size, with_auto_augment=with_auto_augment, random_crop=True)
        temp_trainset = PandasDataSetWithPaths(csvfiles[0], transform=random_transform, ret_path=False)
        datasets.append(temp_trainset)

    trainset = ConcatDataset(datasets)
    if oversample:
        gts = _get_gts(trainset)
        indexes = np.array(list(range(gts.shape[0])))
        ros = RandomOverSampler()
        idxs, labels = ros.fit_resample(indexes.reshape(-1, 1), gts)
        idxs = np.squeeze(idxs).tolist()
        labels = np.squeeze(labels).tolist()
        labels = [int(e) for e in labels]
        num_classes = np.unique(labels).shape[0]

        final_idx = _balance_order(idxs, labels, num_classes, batch_size=train_batch_size)
        sampler = SubsetSequentialSampler(indices=final_idx)

        train_loader = DataLoader(trainset, batch_size=train_batch_size, sampler=sampler, num_workers=16)
    else:
        train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=16)

    val_transform = transforms.Compose([
        OrderedCrops(crop_size=input_size, ncrops=n_val_crops),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])
    valset = PandasDataSetWithPaths(csvfiles[1], transform=val_transform)
    val_loader = DataLoader(valset, batch_size=1)

    _create_gt_csv_file(loader=val_loader, columns=valset.csv_columns, gtFile=gtFile)

    return train_loader, val_loader, valset.csv_columns


class SubsetSequentialSampler(Sampler):

    def __init__(self, indices, shuffle=True):
        self.num_samples = len(indices)
        if shuffle:
            np.random.shuffle(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples


def oversampling_loaders_custom(csvfiles, train_batch_size, val_batch_size, gtFile, load_gts=True):

    training_transform, val_transform = _get_transforms()

    trainset = PandasDataSetWithPaths(csvfiles[0], transform=training_transform, ret_path=False)
    valset = PandasDataSetWithPaths('/raid/ferles/ISIC2019/folds/ValFold1NoPreproc.csv', transform=val_transform, ret_path=False)
    testset = PandasDataSetWithPaths(csvfiles[1], transform=val_transform)

    if load_gts:
        gts = np.load(f'{abs_path}npzs/indexes/isic/gts_no_preproc.npz')['arr_0']
    else:
        gts = _get_gts(trainset)
        np.savez(f'{abs_path}/npzs/indexes/isic/gts_no_preproc.npz', gts)

    indexes = np.array(list(range(gts.shape[0])))
    ros = RandomOverSampler()
    idxs, labels = ros.fit_resample(indexes.reshape(-1, 1), gts)
    idxs = np.squeeze(idxs).tolist()
    labels = np.squeeze(labels).tolist()
    labels = [int(e) for e in labels]
    num_classes = np.unique(labels).shape[0]

    final_idx = _balance_order(idxs, labels, num_classes, batch_size=train_batch_size)
    sampler = SubsetSequentialSampler(indices=final_idx)

    train_loader = DataLoader(trainset, batch_size=train_batch_size, sampler=sampler, num_workers=16)
    val_loader = DataLoader(valset, batch_size=val_batch_size, num_workers=16)
    test_loader = DataLoader(testset, batch_size=val_batch_size, num_workers=16)

    _create_gt_csv_file(loader=test_loader, columns=testset.csv_columns, gtFile=gtFile)

    return train_loader, val_loader, test_loader, testset.csv_columns


def oversampling_loaders_exclude_class_custom_no_gts(csvfiles, train_batch_size, val_batch_size, gtFile, exclude_class, load_gts=True):

    training_transform, val_transform = _get_transforms()

    trainset = PandasDataSetWithPaths(csvfiles[0], transform=training_transform, exclude_class=exclude_class, ret_path=False)
    valset = PandasDataSetWithPaths('/raid/ferles/ISIC2019/folds/ValFold1NoPreproc.csv', transform=val_transform, exclude_class=exclude_class, ret_path=False)
    testset = PandasDataSetWithPaths(csvfiles[1], transform=val_transform, exclude_class=exclude_class)

    if load_gts:
        gts = np.load(f'{abs_path}npzs/indexes/isic/gts_{exclude_class}.npz')['arr_0']
    else:
        gts = _get_gts(trainset)
        np.savez(f'{abs_path}npzs/indexes/isic/gts_{exclude_class}.npz', gts)

    indexes = np.array(list(range(gts.shape[0])))
    ros = RandomOverSampler()
    idxs, labels = ros.fit_resample(indexes.reshape(-1, 1), gts)
    idxs = np.squeeze(idxs).tolist()
    labels = np.squeeze(labels).tolist()
    labels = [int(e) for e in labels]
    num_classes = np.unique(labels).shape[0]

    final_idx = _balance_order(idxs, labels, num_classes, batch_size=train_batch_size)
    sampler = SubsetSequentialSampler(indices=final_idx)

    train_loader = DataLoader(trainset, batch_size=train_batch_size, sampler=sampler, num_workers=16)
    val_loader = DataLoader(valset, batch_size=val_batch_size, num_workers=16)
    test_loader = DataLoader(testset, batch_size=val_batch_size, num_workers=16)

    _create_gt_csv_file(loader=test_loader, columns=testset.csv_columns, gtFile=gtFile)

    return train_loader, val_loader, test_loader, testset.csv_columns


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


def _get_Dermofit_full_loaders(batch_size, input_size=224):

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    in_path, out_path = '/raid/ferles/DermoFit/In', '/raid/ferles/DermoFit/Out'
    in_dataset, out_dataset = ImageFolder(in_path, transform=val_transform), ImageFolder(out_path, transform=val_transform)

    in_loader = DataLoader(in_dataset, batch_size=batch_size)
    out_loader = DataLoader(out_dataset, batch_size=batch_size)

    return in_loader, out_loader


def _get_isic_loader(batch_size, csvfile='/raid/ferles/ISIC2019/Training_paths_and_classes.csv', input_size=224):

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


def _get_dataset(dataset, transforms, test=False):

    transform_train, transform_test = transforms
    if dataset == 'cifar10':
        if not test:
            trainset = torchvision.datasets.CIFAR10(root='/raid/ferles/data', train=True, download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.CIFAR10(root='/raid/ferles/data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(root='/raid/ferles/data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        if not test:
            trainset = torchvision.datasets.CIFAR100(root='/raid/ferles/data', train=True, download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.CIFAR100(root='/raid/ferles/data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(root='/raid/ferles/data', train=False, download=True, transform=transform_test)
    elif dataset=='mnist':
        if not test:
            trainset = torchvision.datasets.MNIST(root='/raid/ferles/data', train=True, download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.MNIST(root='/raid/ferles/data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.MNIST(root='/raid/ferles/data', train=False, download=True, transform=transform_test)
    elif dataset=='fashionmnist':
        if not test:
            trainset = torchvision.datasets.FashionMNIST(root='/raid/ferles/data', train=True, download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.FashionMNIST(root='/raid/ferles/data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.FashionMNIST(root='/raid/ferles/data', train=False, download=True, transform=transform_test)
    elif dataset=='svhn':
        if not test:
            trainset = torchvision.datasets.SVHN(root='/raid/ferles/data', split='train', download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.SVHN(root='/raid/ferles/data', split='train', download=True, transform=transform_test)
        testset = torchvision.datasets.SVHN(root='/raid/ferles/data', split='test', download=True, transform=transform_test)
    elif dataset=='stl':
        if not test:
            trainset = torchvision.datasets.STL10(root='/raid/ferles/data', split='train', download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.STL10(root='/raid/ferles/data', split='train', download=True, transform=transform_test)
        testset = torchvision.datasets.STL10(root='/raid/ferles/data', split='test', download=True, transform=transform_test)
    elif dataset=='tinyimagenet':
        if not test:
            trainset = TinyImageNetDataset(transform=transform_train)
        else:
            trainset = TinyImageNetDataset(transform=transform_test)
        testset = TinyImageNetDataset(train=False, transform=transform_test)
    elif dataset == 'stanforddogs':
        if not test:
            trainset = GenericImageFolderDataset(root='/raid/ferles/Dogs/Stanford/', transform=transform_train)
        else:
            trainset = GenericImageFolderDataset(root='/raid/ferles/Dogs/Stanford/', transform=transform_test)
        testset = GenericImageFolderDataset(root='/raid/ferles/Dogs/Stanford/', train=False, transform=transform_test)
    elif dataset == 'nabirds':
        if not test:
            trainset = GenericImageFolderDataset(root='/raid/ferles/Birds/nabirds/', transform=transform_train)
        else:
            trainset = GenericImageFolderDataset(root='/raid/ferles/Birds/nabirds/', transform=transform_test)
        testset = GenericImageFolderDataset(root='/raid/ferles/Birds/nabirds/', train=False, transform=transform_test)
    else:
        raise NotImplementedError(f'{dataset} not implemented!')

    return trainset, testset


def _get_image_transforms(dataset, resize):

    if 'cifar' in dataset:
        normalize_cifar = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        if resize:
            image_size = 224

            transform_train = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(image_size),
                    transforms.ToTensor(),
                    normalize_cifar,
                ])

            transform_test = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize_cifar,
                ])

        else:

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_cifar])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize_cifar])

    elif dataset != 'fashionmnist':
        if dataset == 'mnist':
            normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))
        elif dataset == 'tinyimagenet':
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        elif dataset == 'svhn':
            normalize = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        elif dataset == 'stl':
            normalize = transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)

        if resize:

            image_size = 224
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

        else:

            transform_train = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ])

    elif dataset=='fashionmnist':

        if resize:

            image_size = 224
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

        else:

            transform_train = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
            ])

    return transform_train, transform_test


def natural_image_loaders(dataset='cifar10', train_batch_size=32, test_batch_size=32, test=False, validation_test_split=0, save_to_pickle=False, pickle_files=None, resize=True):

    transforms = _get_image_transforms(dataset, resize)
    trainset, testset = _get_dataset(dataset, transforms, test)

    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=16)

    if validation_test_split == 0:
        trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=16)
        return trainloader, testloader
    else:
        if pickle_files is None:
            if dataset == 'tinyimagenet':
                gts = trainset.get_targets()
            elif dataset == 'svhn' or dataset == 'stl':
                gts = trainset.labels
            else:
                gts = trainset.targets
            indexes = list(range(trainset.__len__()))

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_test_split, random_state=global_seed)
            trainset_indices, valset_indices = next(iter(splitter.split(indexes, gts)))

            if save_to_pickle:
                with open(f'train_indices_{dataset}.pickle', 'wb') as train_pickle, open(f'val_indices_{dataset}.pickle', 'wb') as val_pickle:
                    pickle.dump(trainset_indices, train_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(valset_indices, val_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            trainpickle, valpickle = pickle_files
            with open(trainpickle, 'rb') as train_pickle, open(valpickle, 'rb') as val_pickle:
                trainset_indices = pickle.load(train_pickle)
                valset_indices = pickle.load(val_pickle)

        train_sampler = SubsetRandomSampler(trainset_indices)
        test_sampler = SubsetRandomSampler(valset_indices)
        if dataset=='svhn':
            trainloader = DataLoader(trainset, batch_size=test_batch_size, sampler=train_sampler, num_workers=16, drop_last=True)
        else:
            trainloader = DataLoader(trainset, batch_size=test_batch_size, sampler=train_sampler, num_workers=16)
        val_loader = DataLoader(trainset, batch_size=test_batch_size, sampler=test_sampler, num_workers=16)

        return trainloader, val_loader, testloader


def _get_fine_grained_transforms():

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(45),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]),

    return transform_train, transform_test


def fine_grained_image_loaders(dataset, train_batch_size=32, test_batch_size=32, test=False, validation_test_split=0, save_to_pickle=False, pickle_files=None, resize=True):

    transforms = _get_fine_grained_transforms()
    trainset, testset = _get_dataset(dataset, transforms, test)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=16)

    if validation_test_split == 0:
        trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=16)
        return trainloader, testloader
    else:
        if pickle_files is None:
            gts = trainset.get_targets()
            indexes = list(range(trainset.__len__()))

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_test_split, random_state=global_seed)
            trainset_indices, valset_indices = next(iter(splitter.split(indexes, gts)))

            if save_to_pickle:
                with open(f'train_indices_{dataset}.pickle', 'wb') as train_pickle, open(f'val_indices_{dataset}.pickle', 'wb') as val_pickle:
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


def create_ensemble_loaders(dataset, num_classes, pickle_files, k=5, train_batch_size=32, test_batch_size=32, test=False, resize=True):

    transforms = _get_image_transforms(dataset, resize)
    trainset, testset = _get_dataset(dataset, transforms, test)
    from copy import deepcopy
    valset = deepcopy(trainset)

    trainpickle, valpickle = pickle_files
    with open(trainpickle, 'rb') as train_pickle, open(valpickle, 'rb') as val_pickle:
        trainset_indices = pickle.load(train_pickle)
        valset_indices = pickle.load(val_pickle)

    unique_labels = list(np.random.permutation(num_classes))
    step = len(unique_labels) // k
    point = 0

    train_ind_loaders, train_ood_loaders = [], []
    val_ind_loaders, val_ood_loaders = [], []
    test_ind_loaders, test_ood_loaders = [], []

    if dataset == 'tinyimagenet':
        gts = trainset.get_targets()
        test_gts = testset.get_targets()
    elif dataset == 'svhn' or dataset == 'stl':
        gts = trainset.labels
        test_gts = testset.labels
    else:
        gts = trainset.targets
        test_gts = testset.targets

    while point < len(unique_labels):

        temp_labels = unique_labels[point: min(len(unique_labels), point+step)]

        custom_trainset_ind = CustomEnsembleDatasetIn(trainset, gts=gts, remove_labels=temp_labels, keep_indices=trainset_indices)
        custom_trainset_out = CustomEnsembleDatasetOut(trainset, gts=gts, remove_labels=temp_labels, keep_indices=trainset_indices)

        train_ind_sampler = SubsetRandomSampler(custom_trainset_ind.keep_indices)
        train_ood_sampler = SubsetRandomSampler(custom_trainset_out.keep_indices)

        train_ind_loader = DataLoader(custom_trainset_ind, batch_size=train_batch_size, sampler=train_ind_sampler)
        train_ood_loader = DataLoader(custom_trainset_out, batch_size=train_batch_size, sampler=train_ood_sampler)

        train_ind_loaders.append(train_ind_loader)
        train_ood_loaders.append(train_ood_loader)

        custom_valset_ind = CustomEnsembleDatasetIn(valset, gts=gts, remove_labels=temp_labels, keep_indices=valset_indices)
        custom_valset_out = CustomEnsembleDatasetOut(valset, gts=gts, remove_labels=temp_labels, keep_indices=valset_indices)

        val_ind_sampler = SubsetRandomSampler(custom_valset_ind.keep_indices)
        val_ood_sampler = SubsetRandomSampler(custom_valset_out.keep_indices)

        val_ind_loader = DataLoader(custom_valset_ind, batch_size=test_batch_size, sampler=val_ind_sampler)
        val_ood_loader = DataLoader(custom_valset_out, batch_size=test_batch_size, sampler=val_ood_sampler)

        val_ind_loaders.append(val_ind_loader)
        val_ood_loaders.append(val_ood_loader)

        testset_indices = list(range(testset.__len__()))
        custom_testset_ind = CustomEnsembleDatasetIn(testset, gts=test_gts, remove_labels=temp_labels, keep_indices=testset_indices)
        custom_testset_out = CustomEnsembleDatasetOut(testset, gts=test_gts, remove_labels=temp_labels, keep_indices=testset_indices)

        test_ind_sampler = SubsetRandomSampler(custom_testset_ind.keep_indices)
        test_ood_sampler = SubsetRandomSampler(custom_testset_out.keep_indices)

        test_ind_loader = DataLoader(custom_testset_ind, batch_size=test_batch_size, sampler=test_ind_sampler)
        test_ood_loader = DataLoader(custom_testset_out, batch_size=test_batch_size, sampler=test_ood_sampler)

        test_ind_loaders.append(test_ind_loader)
        test_ood_loaders.append(test_ood_loader)

        point += step

    return train_ind_loaders, train_ood_loaders, val_ind_loaders, val_ood_loader, test_ind_loaders, test_ood_loaders


def get_ood_detection_data_loaders(ind_dataset, ood_dataset, val_ood_dataset=None, batch_size=32, resize=True):

    natural_datasets = ['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'mnist', 'fashionmnist', 'stl']
    finegrained_datasets = ['stanforddogs', 'nabirds']
    # dermatology_datasets = ['isic', 'dermofit']

    if ind_dataset in natural_datasets:
        transforms = _get_natural_image_transforms(ind_dataset, resize)
    elif ood_dataset in finegrained_datasets:
        transforms = _get_natural_image_transforms(ind_dataset, resize)
    else:
        transforms = _get_transforms()

    if ind_dataset in natural_datasets or finegrained_datasets:
        trainset_ind, testset_ind = _get_dataset(ind_dataset, transforms, test=True)
        with open(f'train_indices_{ind_dataset}.pickle', 'rb') as train_index_pickle, \
                open(f'val_indices_{ind_dataset}.pickle', 'rb') as val_index_pickle:
            train_indexes = pickle.load(train_index_pickle)
            val_indexes = pickle.load(val_index_pickle)
        train_sampler = SubsetRandomSampler(train_indexes)
        val_sampler = SubsetRandomSampler(val_indexes)
        train_loader_ind = DataLoader(trainset_ind, batch_size=batch_size, sampler=train_sampler)
        val_loader_ind = DataLoader(trainset_ind, batch_size=batch_size, sampler=val_sampler)
        test_loader_ind = DataLoader(testset_ind, batch_size=batch_size)
    else:
        #TODO
        pass

    if ood_dataset in natural_datasets or finegrained_datasets:
        trainset_ood, testset_ood = _get_dataset(ood_dataset, transforms, test=True)
        with open(f'val_indices_{ood_dataset}.pickle', 'rb') as index_pickle:
            oodexes = pickle.load(index_pickle)
        val_sampler = SubsetRandomSampler(oodexes)
        val_loader_ood = DataLoader(trainset_ood, batch_size=batch_size, sampler=val_sampler)
        test_loader_ood = DataLoader(testset_ood, batch_size=batch_size)
    else:
        #TODO
        pass

    if val_ood_dataset is not None:
        if val_ood_dataset in natural_datasets or finegrained_datasets:
            trainset_val_ood, testset_val_ood = _get_dataset(val_ood_dataset, transforms, test=True)
            with open(f'val_indices_{val_ood_dataset}.pickle', 'rb') as index_pickle:
                val_oodexes = pickle.load(index_pickle)
            val_sampler = SubsetRandomSampler(val_oodexes)
            val_loader_val_ood = DataLoader(trainset_val_ood, batch_size=batch_size, sampler=val_sampler)
            test_loader_val_ood = DataLoader(testset_val_ood, batch_size=batch_size)
        else:
            #TODO
            pass
        return train_loader_ind, val_loader_ind, test_loader_ind, val_loader_ood, test_loader_ood, val_loader_val_ood, test_loader_val_ood
    else:
        return train_loader_ind, val_loader_ind, test_loader_ind, val_loader_ood, test_loader_ood


def get_temp_data_loaders(batch_size=32):

    ind_dataset = 'cifar10'
    transforms = _get_natural_image_transforms(ind_dataset, resize=False)

    trainset_ind, testset_ind = _get_dataset(ind_dataset, transforms, test=True)
    test_loader_ind = DataLoader(testset_ind, batch_size=batch_size)

    testset = ImageFolder(root='/raid/ferles/ood_benchmark/Imagenet_resize', transform=transforms[1])
    test_loader_ood = DataLoader(testset, batch_size=batch_size)

    return test_loader_ind, test_loader_ood

