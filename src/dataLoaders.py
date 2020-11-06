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

        train_loader = DataLoader(trainset, batch_size=train_batch_size, sampler=sampler, num_workers=3)
    else:
        train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=3)

    val_transform = transforms.Compose([
        OrderedCrops(crop_size=input_size, ncrops=n_val_crops),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])
    valset = PandasDataSetWithPaths(csvfiles[1], transform=val_transform)
    val_loader = DataLoader(valset, batch_size=1)

    # _create_gt_csv_file(loader=val_loader, columns=valset.csv_columns, gtFile=gtFile)

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


def _get_isic_loaders_ood(batch_size,
                          train_csv='./DatasetsISIC2019/folds/Train_Fold_new_no_preproc.csv',
                          val_csv='./DatasetsISIC2019/folds/Val_Fold_new_no_preproc.csv',
                          test_csv='./DatasetsISIC2019/folds/ValFold1NoPreproc.csv',
                          full_csv='./DatasetsISIC2019/Training_paths_and_classes_no_preproc.csv',
                          exclude_class=None):

    if not os.path.exists('/storage/ferles'):
        train_csv = train_csv.replace('storage', 'home')
        val_csv = val_csv.replace('storage', 'home')
        test_csv = test_csv.replace('storage', 'home')
        full_csv = full_csv.replace('storage', 'home')

    input_size = 224
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    trainset = PandasDataSetWithPaths(train_csv, transform=val_transform, exclude_class=exclude_class, ret_path=False)
    valset = PandasDataSetWithPaths(val_csv, transform=val_transform, exclude_class=exclude_class, ret_path=False)
    testset = PandasDataSetWithPaths(test_csv, transform=val_transform, exclude_class=exclude_class, ret_path=False)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=3)
    val_loader = DataLoader(valset, batch_size=batch_size, num_workers=3)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=3)

    if exclude_class is not None:
        ood_set = PandasDataSetSingleClass(full_csv, single_class=exclude_class, transform=val_transform)
        ood_loader = DataLoader(ood_set, batch_size=batch_size, num_workers=3)
        return trainloader, val_loader, testloader, ood_loader
    else:
        return trainloader, val_loader, testloader


def oversampling_loaders_custom(csvfiles, train_batch_size, val_batch_size, gtFile, load_gts=True):

    training_transform, val_transform = _get_transforms()

    if os.path.exists(csvfiles[0]):
        trainset = PandasDataSetWithPaths(csvfiles[0], transform=training_transform, ret_path=False)
        valset = PandasDataSetWithPaths('./DatasetsISIC2019/folds/ValFold1NoPreproc.csv', transform=val_transform, ret_path=False)
        testset = PandasDataSetWithPaths(csvfiles[1], transform=val_transform)
    else:
        trainset = PandasDataSetWithPaths(csvfiles[0].replace('storage', 'home'), transform=training_transform, ret_path=False)
        valset = PandasDataSetWithPaths('/home/ferles/ISIC2019/folds/ValFold1NoPreproc.csv', transform=val_transform, ret_path=False)
        testset = PandasDataSetWithPaths(csvfiles[1].replace('storage', 'home'), transform=val_transform)

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

    train_loader = DataLoader(trainset, batch_size=train_batch_size, sampler=sampler, num_workers=3)
    val_loader = DataLoader(valset, batch_size=val_batch_size, num_workers=3)
    test_loader = DataLoader(testset, batch_size=val_batch_size, num_workers=3)

    # _create_gt_csv_file(loader=test_loader, columns=testset.csv_columns, gtFile=gtFile)

    return train_loader, val_loader, test_loader, testset.csv_columns


def oversampling_loaders_exclude_class_custom_no_gts(csvfiles, train_batch_size, val_batch_size, gtFile, exclude_class, load_gts=True):

    training_transform, val_transform = _get_transforms()

    if os.path.exists(csvfiles[0]):
        trainset = PandasDataSetWithPaths(csvfiles[0], transform=training_transform, exclude_class=exclude_class, ret_path=False)
        valset = PandasDataSetWithPaths('./DatasetsISIC2019/folds/ValFold1NoPreproc.csv', transform=val_transform, exclude_class=exclude_class, ret_path=False)
        testset = PandasDataSetWithPaths(csvfiles[1], transform=val_transform, exclude_class=exclude_class)
    else:
        trainset = PandasDataSetWithPaths(csvfiles[0].replace('storage', 'home'), transform=training_transform, exclude_class=exclude_class, ret_path=False)
        valset = PandasDataSetWithPaths('/home/ferles/ISIC2019/folds/ValFold1NoPreproc.csv', transform=val_transform, exclude_class=exclude_class, ret_path=False)
        testset = PandasDataSetWithPaths(csvfiles[1].replace('storage', 'home'), transform=val_transform, exclude_class=exclude_class)

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

    train_loader = DataLoader(trainset, batch_size=train_batch_size, sampler=sampler, num_workers=3)
    if 'gen' in gtFile.lower() and exclude_class == 'DF':
        val_loader = DataLoader(valset, batch_size=val_batch_size, num_workers=3, drop_last=True)
    else:
        val_loader = DataLoader(valset, batch_size=val_batch_size, num_workers=3)
    test_loader = DataLoader(testset, batch_size=val_batch_size, num_workers=3)

    # _create_gt_csv_file(loader=test_loader, columns=testset.csv_columns, gtFile=gtFile)

    return train_loader, val_loader, test_loader, testset.csv_columns


def _get_7point_loaders(batch_size, csvfile='./Datasets7-point/7pointAsISIC.csv'):

    input_size = 224
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    val_set = PandasDataSetWithPaths(csvfile, transform=val_transform, exclude_class=None, ret_path=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=3)

    return val_loader, val_set.csv_columns


def _get_Dermofit_full_loaders(batch_size, input_size=224):

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    in_path, out_path = './DatasetsDermoFit/In', './DatasetsDermoFit/Out'
    in_dataset, out_dataset = ImageFolder(in_path, transform=val_transform), ImageFolder(out_path, transform=val_transform)

    in_loader = DataLoader(in_dataset, batch_size=batch_size)
    out_loader = DataLoader(out_dataset, batch_size=batch_size)

    return in_loader, out_loader


def _get_isic_loader(batch_size, csvfile='./DatasetsISIC2019/Training_paths_and_classes.csv', input_size=224):

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    val_set = PandasDataSetWithPaths(csvfile, transform=val_transform, exclude_class=None, ret_path=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=3)

    return val_loader


def _get_7point_loaders_ood(batch_size, exclude_class=None, csvfile='./Datasets7-point/7pointAsISIC.csv', out_mode=False):

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

    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=3)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, num_workers=3)

    return val_loader, ood_loader


def _get_custom_loader_7point(batch_size, exclude_class, csvfile='./Datasets7-point/7pointdermClasses.csv'):

    input_size = 224
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    ood_dataset = PandasDataSetSingleClass(csvfile, transform=val_transform, single_class=exclude_class)

    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, num_workers=3)

    return ood_loader, ood_loader


def _get_dataset(dataset, transforms, test=False):

    if os.path.exists('./Datasetsdata'):
        root = './Datasetsdata'
        tiny_imagenet_cifar10_root = './Datasetstiny-imagenet-200/Cifar10'
        tiny_imagenet_cifar100_root = './Datasetstiny-imagenet-200/Cifar100'
        dogs_root = './DatasetsDogs/Stanford/'
        birds_root = './DatasetsBirds/nabirds/'
    else:
        root = '/home/ferles/data'
        tiny_imagenet_cifar10_root = '/home/ferles/tiny-imagenet-200/Cifar10'
        tiny_imagenet_cifar100_root = '/home/ferles/tiny-imagenet-200/Cifar100'
        dogs_root = '/home/ferles/Dogs/Stanford/'
        birds_root = '/home/ferles/Birds/nabirds/'

    transform_train, transform_test = transforms
    if dataset == 'cifar10':
        if not test:
            trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        if not test:
            trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    elif dataset=='mnist':
        if not test:
            trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    elif dataset=='fashionmnist':
        if not test:
            trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform_test)
    elif dataset=='svhn':
        if not test:
            trainset = torchvision.datasets.SVHN(root=root, split='train', download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.SVHN(root=root, split='train', download=True, transform=transform_test)
        testset = torchvision.datasets.SVHN(root=root, split='test', download=True, transform=transform_test)
    elif dataset=='stl':
        if not test:
            trainset = torchvision.datasets.STL10(root=root, split='train', download=True, transform=transform_train)
        else:
            trainset = torchvision.datasets.STL10(root=root, split='train', download=True, transform=transform_test)
        testset = torchvision.datasets.STL10(root=root, split='test', download=True, transform=transform_test)
    elif dataset=='tinyimagenet':
        if not test:
            trainset = TinyImageNetDataset(transform=transform_train)
        else:
            trainset = TinyImageNetDataset(transform=transform_test)
        testset = TinyImageNetDataset(train=False, transform=transform_test)
    elif dataset == 'tinyimagenet-cifar10':
        if not test:
            trainset = GenericImageFolderDataset(root=tiny_imagenet_cifar10_root, type='tiny', transform=transform_train)
        else:
            trainset = GenericImageFolderDataset(root=tiny_imagenet_cifar10_root, type='tiny', transform=transform_test)
        testset = GenericImageFolderDataset(root=tiny_imagenet_cifar10_root, type='tiny', train=False, transform=transform_test)
    elif dataset == 'tinyimagenet-cifar100':
        if not test:
            trainset = GenericImageFolderDataset(root=tiny_imagenet_cifar100_root, type='tiny', transform=transform_train)
        else:
            trainset = GenericImageFolderDataset(root=tiny_imagenet_cifar100_root, type='tiny', transform=transform_test)
        testset = GenericImageFolderDataset(root=tiny_imagenet_cifar100_root, type='tiny', train=False, transform=transform_test)
    elif dataset == 'stanforddogs':
        if not test:
            trainset = GenericImageFolderDataset(root=dogs_root, transform=transform_train)
        else:
            trainset = GenericImageFolderDataset(root=dogs_root, transform=transform_test)
        testset = GenericImageFolderDataset(root=dogs_root, train=False, transform=transform_test)
    elif dataset == 'nabirds':
        if not test:
            trainset = GenericImageFolderDataset(root=birds_root, type='birds', transform=transform_train)
        else:
            trainset = GenericImageFolderDataset(root=birds_root, type='birds', transform=transform_test)
        testset = GenericImageFolderDataset(root=birds_root, type='birds', train=False, transform=transform_test)
    else:
        raise NotImplementedError(f'{dataset} not implemented!')

    return trainset, testset


def _get_subset(dataset, subset_index, transforms, single=False, test=False):

    if os.path.exists('/storage/ferles'):
        if dataset == 'stanforddogs':
            if single:
                    root = f'./DatasetsDogs/Stanford/Subset{subset_index}'
            else:
                root = f'./DatasetsDogs/Stanford/Dataset{subset_index}'
        elif dataset == 'nabirds':
            if single:
                root = f'./DatasetsBirds/nabirds/Subset{subset_index}'
            else:
                root = f'./DatasetsBirds/nabirds/Dataset{subset_index}'
        else:
            raise NotImplementedError(f'{dataset} not implemented!')
    else:
        if dataset == 'stanforddogs':
            if single:
                root = f'/home/ferles/Dogs/Stanford/Subset{subset_index}'
            else:
                root = f'/home/ferles/Dogs/Stanford/Dataset{subset_index}'
        elif dataset == 'nabirds':
            if single:
                root = f'/home/ferles/Birds/nabirds/Subset{subset_index}'
            else:
                root = f'/home/ferles/Birds/nabirds/Dataset{subset_index}'
        else:
            raise NotImplementedError(f'{dataset} not implemented!')

    transform_train, transform_test = transforms
    if not single:
        if dataset == 'stanforddogs':
            if not test:
                trainset = GenericImageFolderDataset(root=root, transform=transform_train, subset_index=subset_index)
            else:
                trainset = GenericImageFolderDataset(root=root, transform=transform_test, subset_index=subset_index)
            testset = GenericImageFolderDataset(root=root, train=False, transform=transform_test, subset_index=subset_index)
        elif dataset == 'nabirds':
            if not test:
                trainset = GenericImageFolderDataset(root=root, type='birds', transform=transform_train, subset_index=subset_index)
            else:
                trainset = GenericImageFolderDataset(root=root, type='birds', transform=transform_test, subset_index=subset_index)
            testset = GenericImageFolderDataset(root=root, type='birds', train=False, transform=transform_test, subset_index=subset_index)
    else:
        if test:
            trainset = ImageFolder(os.path.join(root, 'Train'), transform=transform_test[0])
        else:
            trainset = ImageFolder(os.path.join(root, 'Train'), transform=transform_train[0])
        testset = ImageFolder(os.path.join(root, 'Test'), transform=transform_test[0])
    # ipdb.set_trace()
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
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize_cifar])

    else:
        if dataset == 'tinyimagenet' or dataset == 'tinyimagenet-cifar10' or dataset == 'tinyimagenet-cifar100':
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        elif dataset == 'svhn':
            normalize = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        elif dataset == 'stl':
            normalize = transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)

        if resize:

            image_size = 224
            transform_train = transforms.Compose([
                # transforms.Resize(32),
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

            transform_test = transforms.Compose([
                # transforms.Resize(32),
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

    return transform_train, transform_test


def natural_image_loaders(dataset='cifar10', train_batch_size=32, test_batch_size=32, test=False, validation_test_split=0, save_to_pickle=False, pickle_files=None, resize=True):

    transforms = _get_image_transforms(dataset, resize)
    trainset, testset = _get_dataset(dataset, transforms, test)

    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=3)

    if validation_test_split == 0:
        trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=3)
        return trainloader, testloader
    else:
        if pickle_files is None:
            if dataset == 'tinyimagenet' or dataset == 'tinyimagenet-cifar10' or dataset == 'tinyimagenet-cifar100':
                gts = trainset.get_targets()
            elif dataset == 'svhn' or dataset == 'stl':
                gts = trainset.labels
            else:
                gts = trainset.targets
            indexes = list(range(trainset.__len__()))

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_test_split, random_state=global_seed)
            trainset_indices, valset_indices = next(iter(splitter.split(indexes, gts)))
            if dataset == 'tinyimagenet' or dataset == 'tinyimagenet-cifar10' or dataset == 'tinyimagenet-cifar100':
                random.shuffle(trainset_indices)
            if save_to_pickle:
                with open(f'pickle_files/train_indices_{dataset}.pickle', 'wb') as train_pickle, open(f'pickle_files/val_indices_{dataset}.pickle', 'wb') as val_pickle:
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
            trainloader = DataLoader(trainset, batch_size=test_batch_size, sampler=train_sampler, num_workers=3, drop_last=True)
        else:
            trainloader = DataLoader(trainset, batch_size=test_batch_size, sampler=train_sampler, num_workers=3)
        val_loader = DataLoader(trainset, batch_size=test_batch_size, sampler=test_sampler, num_workers=3)

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
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=3)

    if validation_test_split == 0:
        trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=3)
        return trainloader, testloader
    else:
        if pickle_files is None:
            gts = trainset.get_targets()
            indexes = list(range(trainset.__len__()))

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_test_split, random_state=global_seed)
            trainset_indices, valset_indices = next(iter(splitter.split(indexes, gts)))

            if save_to_pickle:
                with open(f'pickle_files/train_indices_{dataset}.pickle', 'wb') as train_pickle, open(f'pickle_files/val_indices_{dataset}.pickle', 'wb') as val_pickle:
                    pickle.dump(trainset_indices, train_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(valset_indices, val_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            trainpickle, valpickle = pickle_files
            with open(trainpickle, 'rb') as train_pickle, open(valpickle, 'rb') as val_pickle:
                trainset_indices = pickle.load(train_pickle)
                valset_indices = pickle.load(val_pickle)

        train_sampler = SubsetRandomSampler(trainset_indices)
        test_sampler = SubsetRandomSampler(valset_indices)
        trainloader = DataLoader(trainset, batch_size=test_batch_size, sampler=train_sampler, num_workers=3)
        val_loader = DataLoader(trainset, batch_size=test_batch_size, sampler=test_sampler, num_workers=3)

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
    num_classes = []
    dicts = []

    if 'tinyimagenet' in dataset:
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
        in_labels = list(set(unique_labels) - set(temp_labels))
        in_gts_indexes = [index for index in list(range(len(gts))) if gts[index] not in temp_labels]
        out_gts_indexes = [index for index in list(range(len(gts))) if gts[index] in temp_labels]

        train_in_indices = list(set(trainset_indices).intersection(set(in_gts_indexes)))
        train_out_indices = list(set(trainset_indices).intersection(set(out_gts_indexes)))

        val_in_indices = list(set(valset_indices).intersection(set(in_gts_indexes)))

        num_classes.append(len(set(gts) - set(temp_labels)))
        temp = zip(in_labels, list(range(len(in_labels))))
        dicts.append(dict(temp))

        train_ind_sampler = SubsetRandomSampler(train_in_indices)
        train_ood_sampler = SubsetRandomSampler(train_out_indices)

        train_ind_loader = DataLoader(trainset, batch_size=train_batch_size, sampler=train_ind_sampler, num_workers=8)
        train_ood_loader = DataLoader(trainset, batch_size=train_batch_size, sampler=train_ood_sampler, num_workers=8)

        train_ind_loaders.append(train_ind_loader)
        train_ood_loaders.append(train_ood_loader)

        val_ind_sampler = SubsetRandomSampler(val_in_indices)
        val_ind_loader = DataLoader(trainset, batch_size=test_batch_size, sampler=val_ind_sampler, num_workers=8)

        val_ind_loaders.append(val_ind_loader)

        test_in_gts_indexes = [index for index in list(range(len(test_gts))) if test_gts[index] not in temp_labels]
        test_ind_sampler = SubsetRandomSampler(test_in_gts_indexes)
        test_ind_loader = DataLoader(testset, batch_size=test_batch_size, sampler=test_ind_sampler, num_workers=8)

        test_ind_loaders.append(test_ind_loader)
        point += step

    return train_ind_loaders, train_ood_loaders, val_ind_loaders, test_ind_loaders, num_classes, dicts


def fine_grained_image_loaders_subset(dataset, subset_index, single=False, train_batch_size=32, test_batch_size=32, test=False, validation_test_split=0, save_to_pickle=False, pickle_files=None, ret_num_classes=False):

    transforms = _get_fine_grained_transforms()
    trainset, testset = _get_subset(dataset, subset_index, transforms, single, test)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=3)

    if validation_test_split == 0:
        trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=3)
        return trainloader, testloader
    else:
        if pickle_files is None:
            gts = trainset.get_targets()
            indexes = list(range(trainset.__len__()))

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_test_split, random_state=global_seed)
            trainset_indices, valset_indices = next(iter(splitter.split(indexes, gts)))

            # if save_to_pickle:
            with open(f'pickle_files/train_indices_{dataset}_subset_{subset_index}.pickle', 'wb') as train_pickle, open(f'pickle_files/val_indices_{dataset}_subset_{subset_index}.pickle', 'wb') as val_pickle:
                pickle.dump(trainset_indices, train_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(valset_indices, val_pickle, protocol=pickle.HIGHEST_PROTOCOL)

            train_sampler = SubsetRandomSampler(trainset_indices)
            test_sampler = SubsetRandomSampler(valset_indices)
            trainloader = DataLoader(trainset, batch_size=train_batch_size, sampler=train_sampler, num_workers=3)
            val_loader = DataLoader(trainset, batch_size=test_batch_size, sampler=test_sampler, num_workers=3)

            if ret_num_classes:
                return trainloader, val_loader, testloader, trainset.num_classes
            else:
                return trainloader, val_loader, testloader
        else:
            if not single:
                trainpickle, valpickle = pickle_files
                with open(trainpickle, 'rb') as train_pickle, open(valpickle, 'rb') as val_pickle:
                    trainset_indices = pickle.load(train_pickle)
                    valset_indices = pickle.load(val_pickle)

                train_sampler = SubsetRandomSampler(trainset_indices)
                test_sampler = SubsetRandomSampler(valset_indices)
                trainloader = DataLoader(trainset, batch_size=train_batch_size, sampler=train_sampler, num_workers=3)
                val_loader = DataLoader(trainset, batch_size=test_batch_size, sampler=test_sampler, num_workers=3)

                if ret_num_classes:
                    return trainloader, val_loader, testloader, trainset.num_classes
                else:
                    return trainloader, val_loader, testloader

            else:
                trainloader = DataLoader(trainset, batch_size=train_batch_size, num_workers=3)
                return trainloader


def get_ood_loaders(ind_dataset, val_ood_dataset, test_ood_dataset, batch_size=32, dataset_size=1000, exclude_class=None, subset_index=None):

    val_batch_size = batch_size

    if ind_dataset == 'isic':
        _, transform_test = _get_transforms()
    elif ind_dataset not in ['stanforddogs', 'nabirds']:
        _, transform_test = _get_image_transforms(ind_dataset, resize=True)
    else:
        _, transform_test = _get_fine_grained_transforms()

    if val_ood_dataset == '7point':
        dataset_size = 43
    elif val_ood_dataset == 'soodogs':
        dataset_size = 897

    if ind_dataset == 'isic':
        if os.path.exists('/storage/ferles'):
            path = './DatasetsISIC2019/folds/'
        else:
            path = '/home/ferles/ISIC2019/folds/'

        print(exclude_class)
        ind_trainset = PandasDataSetWithPaths(f'{path}Train_Fold_new_no_preproc.csv', transform=transform_test, exclude_class=exclude_class, ret_path=False)
        ind_valset = PandasDataSetWithPaths(f'{path}ValFold1NoPreproc.csv', transform=transform_test, exclude_class=exclude_class, ret_path=False)
        ind_testset = PandasDataSetWithPaths(f'{path}Val_Fold_new_no_preproc.csv', transform=transform_test, exclude_class=exclude_class, ret_path=False)

        if ind_valset.__len__() < dataset_size:
            dataset_size = ind_valset.__len__()
        indexes = list(range(ind_valset.__len__()))
        random.shuffle(indexes)
        indexes = indexes[:dataset_size]
        val_ind_sampler = SubsetRandomSampler(indexes)

        train_ind_loader = DataLoader(ind_trainset, batch_size=batch_size, num_workers=3)
        val_ind_loader = DataLoader(ind_valset, batch_size=val_batch_size, sampler=val_ind_sampler, num_workers=3)
        test_ind_loader = DataLoader(ind_testset, batch_size=batch_size, num_workers=3)
    elif ind_dataset == 'stanforddogs' or ind_dataset == 'nabirds':
        if subset_index is None:
            ind_trainset, ind_testset = _get_dataset(ind_dataset, [transform_test, transform_test], test=True)
            with open(f'pickle_files/train_indices_{ind_dataset}.pickle', 'rb') as train_pickle, open(f'pickle_files/val_indices_{ind_dataset}.pickle', 'rb') as val_pickle:
                # print(f'train_indices_{ind_dataset}.pickle')
                # print(f'val_indices_{ind_dataset}.pickle')
                trainset_indices = pickle.load(train_pickle)
                valset_indices = pickle.load(val_pickle)

                train_sampler = SubsetRandomSampler(trainset_indices)
                val_sampler = SubsetRandomSampler(valset_indices)
                train_ind_loader = DataLoader(ind_trainset, batch_size=batch_size, num_workers=3, sampler=train_sampler)
                val_ind_loader = DataLoader(ind_trainset, batch_size=val_batch_size, num_workers=3, sampler=val_sampler)
                test_ind_loader = DataLoader(ind_testset, batch_size=batch_size, num_workers=3)
        else:
            ind_trainset, ind_testset = _get_subset(ind_dataset, subset_index, [transform_test, transform_test], test=True)
            with open(f'pickle_files/train_indices_{ind_dataset}_subset_{subset_index}.pickle', 'rb') as train_pickle, open(f'pickle_files/val_indices_{ind_dataset}_subset_{subset_index}.pickle', 'rb') as val_pickle:
                trainset_indices = pickle.load(train_pickle)
                valset_indices = pickle.load(val_pickle)

                train_sampler = SubsetRandomSampler(trainset_indices)
                val_sampler = SubsetRandomSampler(valset_indices)
                train_ind_loader = DataLoader(ind_trainset, batch_size=batch_size, num_workers=3, sampler=train_sampler)
                val_ind_loader = DataLoader(ind_trainset, batch_size=val_batch_size, num_workers=3, sampler=val_sampler)
                test_ind_loader = DataLoader(ind_testset, batch_size=batch_size, num_workers=3)
    elif ind_dataset in ['cifar10', 'cifar100', 'svhn', 'stl', 'tinyimagenet']:
        ind_trainset, ind_testset = _get_dataset(ind_dataset, [transform_test, transform_test], test=True)
        with open(f'pickle_files/train_indices_{ind_dataset}.pickle', 'rb') as train_pickle, open(f'pickle_files/val_indices_{ind_dataset}.pickle', 'rb') as val_pickle:
            trainset_indices = pickle.load(train_pickle)
            valset_indices = pickle.load(val_pickle)

            train_sampler = SubsetRandomSampler(trainset_indices)
            val_sampler = SubsetRandomSampler(valset_indices)
            train_ind_loader = DataLoader(ind_trainset, batch_size=batch_size, num_workers=3, sampler=train_sampler)
            val_ind_loader = DataLoader(ind_trainset, batch_size=val_batch_size, num_workers=3, sampler=val_sampler)
            test_ind_loader = DataLoader(ind_testset, batch_size=batch_size, num_workers=3)

    if val_ood_dataset != 'fgsm':
        if val_ood_dataset == 'isic':
            if os.path.exists('/storage/ferles'):
                path = './DatasetsISIC2019/folds/'
            else:
                path = '/home/ferles/ISIC2019/folds/'
            if exclude_class is None:
                val_ood_valset = PandasDataSetWithPaths(f'{path}ValFold1NoPreproc.csv', transform=transform_test, exclude_class=exclude_class, ret_path=False)
            else:
                val_ood_valset = PandasDataSetSingleClass(f'{path}Train_Fold_new_no_preproc.csv', transform=transform_test, single_class=exclude_class)
            indexes = list(range(val_ood_valset.__len__()))
            random.shuffle(indexes)
            sampler = SubsetRandomSampler(indexes[:min(len(indexes), dataset_size)])
            val_ood_loader = DataLoader(val_ood_valset, batch_size=batch_size, sampler=sampler, num_workers=3)
        elif val_ood_dataset == 'stanforddogs' or val_ood_dataset == 'nabirds':
            if subset_index is None:
                val_ood_trainset, val_ood_testset = _get_dataset(val_ood_dataset, [transform_test, transform_test], test=True)
                with open(f'pickle_files/train_indices_{val_ood_dataset}.pickle', 'wb') as train_pickle, open(f'pickle_files/val_indices_{val_ood_dataset}.pickle', 'wb') as val_pickle:
                    valset_indices = pickle.load(val_pickle)
                    val_sampler = SubsetRandomSampler(valset_indices)
                    val_ood_loader = DataLoader(val_ood_trainset, batch_size=batch_size, num_workers=3, sampler=val_sampler)
            else:
                val_trainset, _ = _get_subset(ind_dataset, subset_index, [transform_test, transform_test], single=True, test=True)
                indexes = list(range(val_trainset.__len__()))
                random.shuffle(indexes)
                indexes = indexes[:dataset_size]
                val_sampler = SubsetRandomSampler(indexes)
                val_ood_loader = DataLoader(val_trainset, batch_size=val_batch_size, num_workers=3, sampler=val_sampler)
        elif val_ood_dataset in ['cifar10', 'cifar100', 'svhn', 'stl', 'tinyimagenet']:
            val_ood_trainset, val_ood_testset = _get_dataset(val_ood_dataset, [transform_test, transform_test], test=True)
            with open(f'pickle_files/val_indices_{val_ood_dataset}.pickle', 'rb') as val_pickle:
                valset_val_oodices = pickle.load(val_pickle)
                val_sampler = SubsetRandomSampler(valset_val_oodices)
                val_ood_loader = DataLoader(val_ood_trainset, batch_size=batch_size, num_workers=3, sampler=val_sampler)
        elif val_ood_dataset == '7point':
            if os.path.exists('/storage/ferles'):
                path_7_points = './Datasets7-point/non_overlapping/'
            else:
                path_7_points = '/home/ferles/7-point/non_overlapping/'
            dataset7point = ImageFolder(path_7_points, transform=transform_test)
            indexes = list(range(dataset7point.__len__()))
            random.shuffle(indexes)
            indexes = indexes[:min(dataset_size, len(indexes))]
            val_sampler = SubsetRandomSampler(indexes)
            val_ood_loader = DataLoader(dataset7point, batch_size=batch_size, sampler=val_sampler, num_workers=3)
        elif val_ood_dataset == 'cifar10dogs':
            valset = _get_dataset(dataset='cifar10', transforms=[transform_test, transform_test], test=True)
            if not os.path.exists('cifar10dogsindices.pickle'):
                targets = np.array(valset.targets)
                pos = np.where(targets == 5).tolist()
                random.shuffle(pos)
                valset_indices = pos[:dataset_size]
                pickle.dump(valset_indices, open('cifar10dogsindices.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                valset_indices = pickle.load(open('cifar10dogsindices.pickle', 'rb'))
            sampler = SubsetRandomSampler(valset_indices)
            val_ood_loader = DataLoader(valset, batch_size=32, sampler=sampler, num_workers=3)
        elif val_ood_dataset == 'birdsnap':
            if os.path.exists('/storage/ferles'):
                birds_path = './DatasetsBirds/temp_birdsnap/OutBirdsNap'
            else:
                birds_path = '/home/ferles/Birds/temp_birdsnap/OutBirdsNap/'
            dataset_birdsnap = ImageFolder(birds_path, transform=transform_test[0])
            val_ood_loader = DataLoader(dataset_birdsnap, batch_size=batch_size, num_workers=3)
        elif val_ood_dataset == 'imagenet':
            if os.path.exists('/storage/ferles'):
                val_imagenet_path = './DatasetsImageNetVal/'
            else:
                val_imagenet_path = '/home/ferles/ImageNetVal/'
            if ind_dataset == 'stanforddogs' or ind_dataset == 'nabirds':
                dataset_imagenet_val = ImageFolder(val_imagenet_path, transform=transform_test[0])
            else:
                dataset_imagenet_val = ImageFolder(val_imagenet_path, transform=transform_test)
            val_ood_loader = DataLoader(dataset_imagenet_val, batch_size=batch_size, num_workers=3)
        elif val_ood_dataset == 'soodogs':
            if os.path.exists('/storage/ferles'):
                val_imagenet_path = './DatasetsDogs/SOODogs/'
            else:
                val_imagenet_path = '/home/ferles/Dogs/SOODogs/'
            if ind_dataset == 'stanforddogs' or ind_dataset == 'nabirds':
                dataset_imagenet_val = ImageFolder(val_imagenet_path, transform=transform_test[0])
            else:
                dataset_imagenet_val = ImageFolder(val_imagenet_path, transform=transform_test)
            val_ood_loader = DataLoader(dataset_imagenet_val, batch_size=batch_size, num_workers=3)

    if test_ood_dataset == 'isic':
        if os.path.exists('/storage/ferles'):
            path = './DatasetsISIC2019/'
        else:
            path = '/home/ferles/ISIC2019/'
        test_ood_testset = PandasDataSetSingleClass(f'{path}Training_paths_and_classes_no_preproc.csv', transform=transform_test, single_class=exclude_class)
        test_ood_loader = DataLoader(test_ood_testset, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset=='stanforddogs' or test_ood_dataset=='nabirds':
        if subset_index is None:
            _, test_ood_testset = _get_dataset(test_ood_dataset, [transform_test, transform_test], test=True)
            test_ood_loader = DataLoader(test_ood_testset, batch_size=batch_size, num_workers=3)
        else:
            _, test_ood_testset = _get_subset(ind_dataset, subset_index, [transform_test, transform_test], single=True, test=True)
            test_ood_loader = DataLoader(test_ood_testset, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset in ['cifar10', 'cifar100', 'svhn', 'stl', 'tinyimagenet']:
        _, test_ood_testset = _get_dataset(test_ood_dataset, [transform_test, transform_test], test=True)
        test_ood_loader = DataLoader(test_ood_testset, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset == 'oxfordpets':
        if os.path.exists('/storage/ferles'):
            oxford_pets_path = './DatasetsDogs/Oxford/images/'
        else:
            oxford_pets_path = '/home/ferles/Dogs/Oxford/images/'
        if ind_dataset == 'stanforddogs' or ind_dataset == 'nabirds':
            dataset_oxford_pets = ImageFolder(oxford_pets_path, transform=transform_test[0])
        else:
            dataset_oxford_pets = ImageFolder(oxford_pets_path, transform=transform_test[0])
        test_ood_loader = DataLoader(dataset_oxford_pets, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset == 'oxfordpets-in':
        if os.path.exists('/storage/ferles'):
            oxford_pets_path = './DatasetsDogs/Oxford/In/'
        else:
            oxford_pets_path = '/home/ferles/Dogs/Oxford/In/'
        dataset_oxford_pets = ImageFolder(oxford_pets_path, transform=transform_test[0])
        test_ood_loader = DataLoader(dataset_oxford_pets, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset == 'oxfordpets-out':
        if os.path.exists('/storage/ferles'):
            oxford_pets_path = './DatasetsDogs/Oxford/Out/'
        else:
            oxford_pets_path = '/home/ferles/Dogs/Oxford/Out/'
        dataset_oxford_pets = ImageFolder(oxford_pets_path, transform=transform_test[0])
        test_ood_loader = DataLoader(dataset_oxford_pets, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset == 'dermofit':
        if os.path.exists('/storage/ferles'):
            dermofit_path = './DatasetsDermoFit/'
        else:
            dermofit_path = '/home/ferles/DermoFit/'
        dataset_dermofit = ImageFolder(dermofit_path, transform=transform_test)
        test_ood_loader = DataLoader(dataset_dermofit, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset == 'dermofit-in':
        if os.path.exists('/storage/ferles'):
            dermofit_path = './DatasetsDermoFitIn/'
        else:
            dermofit_path = '/home/ferles/DermoFitIn/'
        dataset_dermofit = ImageFolder(dermofit_path, transform=transform_test)
        test_ood_loader = DataLoader(dataset_dermofit, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset == 'dermofit-out':
        if os.path.exists('/storage/ferles'):
            dermofit_path = './DatasetsDermoFitOut/'
        else:
            dermofit_path = '/home/ferles/DermoFitOut/'
        dataset_dermofit = ImageFolder(dermofit_path, transform=transform_test)
        test_ood_loader = DataLoader(dataset_dermofit, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset == 'cub200':
        if os.path.exists('/storage/ferles'):
            cub_path = './DatasetsBirds/CUB200/images/'
        else:
            cub_path = './DatasetsBirds/CUB200/images/'
        dataset_cub = ImageFolder(cub_path, transform=transform_test[0])
        test_ood_loader = DataLoader(dataset_cub, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset == 'cub200-in':
        if os.path.exists('/storage/ferles'):
            cub_path = './DatasetsBirds/CUB200/In/'
        else:
            cub_path = './DatasetsBirds/CUB200/In/'
        dataset_cub = ImageFolder(cub_path, transform=transform_test[0])
        test_ood_loader = DataLoader(dataset_cub, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset == 'cub200-out':
        if os.path.exists('/storage/ferles'):
            cub_path = './DatasetsBirds/CUB200/Out/'
        else:
            cub_path = '/home/ferles/Birds/CUB200/Out/'
        dataset_cub = ImageFolder(cub_path, transform=transform_test[0])
        test_ood_loader = DataLoader(dataset_cub, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset == 'birdsnap':
        if os.path.exists('/storage/ferles'):
            birdsnap_path = './DatasetsBirds/birdsnap/'
        else:
            birdsnap_path = '/home/ferles/Birds/birdsnap'
        dataset_birdsnap = ImageFolder(birdsnap_path, transform=transform_test[0])
        test_ood_loader = DataLoader(dataset_birdsnap, batch_size=batch_size, num_workers=3)
    elif test_ood_dataset == 'places':
        if os.path.exists('/storage/ferles'):
            places_path = './DatasetsPlaces'
        else:
            places_path = '/home/ferles/Places'
        if ind_dataset == 'stanforddogs' or ind_dataset=='nabirds':
            dataset_places = ImageFolder(places_path, transform=transform_test[0])
        else:
            dataset_places = ImageFolder(places_path, transform=transform_test)
        test_ood_loader = DataLoader(dataset_places, batch_size=batch_size, num_workers=3)

    if val_ood_dataset != 'fgsm':
        return train_ind_loader, val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader
    else:
        return [train_ind_loader, val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader]


def get_triplets_loaders(ind_dataset, val_ood_dataset, ood_datasets, batch_size=32, resize=True):

    _, transform_test = _get_image_transforms(ind_dataset, resize=resize)
    ind_trainset, ind_testset = _get_dataset(ind_dataset, transforms=[transform_test, transform_test], test=True)
    with open(f'pickle_files/train_indices_{ind_dataset}.pickle', 'rb') as train_pickle, open(f'pickle_files/val_indices_{ind_dataset}.pickle', 'rb') as val_pickle:
        trainset_indices = pickle.load(train_pickle)
        valset_indices = pickle.load(val_pickle)

        train_sampler = SubsetRandomSampler(trainset_indices)
        val_sampler = SubsetRandomSampler(valset_indices)
        train_ind_loader = DataLoader(ind_trainset, batch_size=batch_size, num_workers=3, sampler=train_sampler)
        val_ind_loader = DataLoader(ind_trainset, batch_size=batch_size, num_workers=3, sampler=val_sampler)
        test_ind_loader = DataLoader(ind_testset, batch_size=batch_size, num_workers=3)

    with open(f'pickle_files/val_indices_{val_ood_dataset}.pickle', 'rb') as val_val_pickle:
        val_ood_trainset, _ = _get_dataset(val_ood_dataset, transforms=[transform_test, transform_test], test=True)
        val_valset_indices = pickle.load(val_val_pickle)

        val_sampler = SubsetRandomSampler(val_valset_indices)
        val_ood_loader = DataLoader(val_ood_trainset, batch_size=batch_size, num_workers=3, sampler=val_sampler)

    _, test_ood_testset_1 = _get_dataset(ood_datasets[0], transforms=[transform_test, transform_test], test=True)
    _, test_ood_testset_2 = _get_dataset(ood_datasets[1], transforms=[transform_test, transform_test], test=True)
    _, test_ood_testset_3 = _get_dataset(ood_datasets[2], transforms=[transform_test, transform_test], test=True)

    test_ood_loader_1 = DataLoader(test_ood_testset_1, batch_size=batch_size, num_workers=3)
    test_ood_loader_2 = DataLoader(test_ood_testset_2, batch_size=batch_size, num_workers=3)
    test_ood_loader_3 = DataLoader(test_ood_testset_3, batch_size=batch_size, num_workers=3)

    return train_ind_loader, val_ind_loader, test_ind_loader, val_ood_loader, test_ood_loader_1, test_ood_loader_2, test_ood_loader_3


def imageNetLoader(dataset, batch_size=32):

    if dataset=='stanforddogs' or dataset == 'nabirds':
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize((0.6796, 0.5284, 0.5193), (0.1200, 0.1413, 0.1538))

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    dataset = MyImageFolder(root='./DatasetsImageNet/', transform=test_transform)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=3, shuffle=True)

    return loader


def tinyImageNetLoader(dataset, batch_size=32):

    if dataset=='stanforddogs' or dataset == 'nabirds':
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
      normalize = transforms.Normalize((0.6796, 0.5284, 0.5193), (0.1200, 0.1413, 0.1538))

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    dataset = MyImageFolder(root='./Datasetstiny-imagenet-200/train', transform=test_transform)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=3)

    return loader


