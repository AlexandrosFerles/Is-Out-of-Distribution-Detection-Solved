import torch
from torch.utils import data
import torchvision
import numpy as np
from PIL import Image
import pandas as pd
import random
from glob import glob as glob
import os
import sys
import pickle
import ipdb


global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


# TODO: MetaData
class PandasDataSet(data.Dataset):

    def __init__(self, csvfile, transform, exclude_class=None):

        self.df = pd.read_csv(csvfile)
        if exclude_class is not None:
            self.df = self.df[self.df[exclude_class] != 1.0]
            self.class_names = list(self.df.columns.values)[1:-1]
            self.class_names.remove(exclude_class)
            self.df = self.df.drop([exclude_class], axis=1)
        else:
            self.class_names = list(self.df.columns.values)[1:-1]
        self.image_path = self.df['image'].tolist()
        self.transform = transform

    def __getitem__(self, index):

        single_image = self.get_x(index)
        gts = self.get_y(index)
        single_image = self.transform(single_image)

        return single_image, gts

    def get_x(self, idx):

        img = Image.open(self.image_path[idx])
        return img

    def get_y(self, idx):
        return torch.IntTensor(self.df.iloc[idx, 1:-1].tolist())

    def __len__(self):
        return len(self.image_path)


class PandasDataSetSingleClass(data.Dataset):

    def __init__(self, csvfile, transform, single_class):
        self.df = pd.read_csv(csvfile)
        self.df = self.df[self.df[single_class] == 1.0]
        self.class_names = [single_class]
        self.df = self.df[['image', single_class]]
        self.image_path = self.df['image'].tolist()
        self.transform = transform

    def __getitem__(self, index):
        single_image = self.get_x(index)
        gts = self.get_y(index)
        single_image = self.transform(single_image)

        return single_image, gts

    def get_x(self, idx):

        img = Image.open(self.image_path[idx])
        return img

    def get_y(self, idx):
        return torch.IntTensor(self.df.iloc[idx, 1:-1].tolist())

    def __len__(self):
        return len(self.image_path)


class PandasDataSetWithPaths(data.Dataset):

    def __init__(self, csvfile, transform, exclude_class=None, ret_path=True):
        self.df = pd.read_csv(csvfile)
        self.ret_path = ret_path
        if exclude_class is not None:
            self.df = self.df[self.df[exclude_class] !=1.0]
            self.class_names = list(self.df.columns.values)[1:-1]
            self.class_names.remove(exclude_class)
            self.csv_columns = list(self.df.columns.values)[:-1]
            self.csv_columns.remove(exclude_class)
            self.df = self.df.drop([exclude_class], axis=1)
        else:
            self.class_names = list(self.df.columns.values)[1:-1]
            self.csv_columns = list(self.df.columns.values)[:-1]
        self.image_path = self.df['image'].tolist()
        self.transform = transform

    def __getitem__(self, index):
        path, single_image = self.get_x(index)
        gts = self.get_y(index)
        single_image = self.transform(single_image)

        if self.ret_path:
            return path, single_image, gts
        else:
            return single_image, gts

    def get_x(self, idx):

        img = Image.open(self.image_path[idx])
        return self.image_path[idx], img

    def get_y(self, idx):
        return torch.IntTensor(self.df.iloc[idx, 1:-1].tolist())

    def __len__(self):
        return len(self.image_path)


class PandasDataSetWithPaths7point(data.Dataset):

    def __init__(self, csvfile, transform, exclude_class=None):
        self.df = pd.read_csv(csvfile)
        if exclude_class is not None:
            self.df = self.df[self.df[exclude_class] !=1.0]
            self.class_names = list(self.df.columns.values)[1:]
            self.class_names.remove(exclude_class)
            self.csv_columns = list(self.df.columns.values)
            self.csv_columns.remove(exclude_class)
            self.df = self.df.drop([exclude_class], axis=1)
        else:
            self.class_names = list(self.df.columns.values)[1:]
            self.csv_columns = list(self.df.columns.values)
        self.image_path = self.df['image'].tolist()
        self.transform = transform

    def __getitem__(self, index):
        path, single_image = self.get_x(index)
        gts = self.get_y(index)
        single_image = self.transform(single_image)

        return path, single_image, gts

    def get_x(self, idx):

        try:
            img = Image.open(self.image_path[idx])
        except IndexError:
            ipdb.set_trace()
        return self.image_path[idx], img

    def get_y(self, idx):
        return torch.IntTensor(self.df.iloc[idx, 1:].tolist())

    def __len__(self):
        return len(self.image_path)


class PandasDataSetWithPathsSingleClass(data.Dataset):

    def __init__(self, csvfile, transform, single_class):
        self.df = pd.read_csv(csvfile)
        self.df = self.df[self.df[single_class] == 1.0]
        self.class_names = [single_class]
        self.df = self.df[['image', single_class]]
        self.image_path = self.df['image'].tolist()
        self.transform = transform

    def __getitem__(self, index):
        path, single_image = self.get_x(index)
        gts = self.get_y(index)
        single_image = self.transform(single_image)

        return path, single_image, gts

    def get_x(self, idx):

        img = Image.open(self.image_path[idx])
        return self.image_path[idx], img

    def get_y(self, idx):
        return torch.IntTensor(self.df.iloc[idx, 1:-1].tolist())

    def __len__(self):
        return len(self.image_path)


class PandasDataSetWithPathsExcludes(data.Dataset):

    def __init__(self, csvfile, transform, exclude_classes, ret_path=True):
        self.ret_path=ret_path
        self.df = pd.read_csv(csvfile)
        class1, class2 = exclude_classes
        self.df = self.df[(self.df[class1] != 1.0) & (self.df[class2] != 1.0)]
        self.class_names = list(self.df.columns.values)[1:]
        self.class_names.remove(class1)
        self.class_names.remove(class2)
        self.csv_columns = list(self.df.columns.values)
        self.csv_columns.remove(class1)
        self.csv_columns.remove(class2)
        self.df = self.df.drop([class1], axis=1)
        self.df = self.df.drop([class2], axis=1)
        self.image_path = self.df['image'].tolist()
        self.transform = transform

    def __getitem__(self, index):
        if self.ret_path:
            path, single_image = self.get_x(index)
        else:
            single_image = self.get_x(index)
        gts = self.get_y(index)
        single_image = self.transform(single_image)

        if self.ret_path:
            return path, single_image, gts
        else:
            return single_image, gts

    def get_x(self, idx):

        img = Image.open(self.image_path[idx])
        if self.ret_path:
            return self.image_path[idx], img
        else:
            return img

    def get_y(self, idx):
        return torch.IntTensor(self.df.iloc[idx, 1:-1].tolist())

    def __len__(self):
        return len(self.image_path)


class PandasDataSetWithPathsSelectClasses(data.Dataset):

    def __init__(self, csvfile, transform, classes, ret_path=True):
        self.ret_path = ret_path
        self.df = pd.read_csv(csvfile)
        class1, class2 = classes
        self.df = self.df[(self.df[class1] == 1.0) | (self.df[class2] == 1.0)]
        self.df = self.df.reset_index()
        self.df = self.df.drop('index', axis=1)
        self.image_path = self.df['image'].tolist()
        self.transform = transform

    def __getitem__(self, index):
        if self.ret_path:
            path, single_image = self.get_x(index)
        else:
            single_image = self.get_x(index)
        gts = self.get_y(index)
        single_image = self.transform(single_image)

        if self.ret_path:
            return path, single_image, gts
        else:
            return single_image, gts

    def get_x(self, idx):

        img = Image.open(self.image_path[idx])
        if self.ret_path:
            return self.image_path[idx], img
        else:
            return img

    def get_y(self, idx):
        return torch.IntTensor(self.df.iloc[idx, 1:-1].tolist())

    def __len__(self):
        return len(self.image_path)


#TODO: Further adapt
class PandasDataSetCustomWithPaths(data.Dataset):

    def __init__(self, csvfile, transform):
        self.df = pd.read_csv(csvfile)
        self.class_names = list(self.df.columns.values)[3:-1]
        self.metadata_columns = list(self.df.columns.values)[1:4]
        # TODO: Confirm that this is correct
        self.csv_columns = list(self.df.columns.values)[0] + self.class_names
        self.image_path = self.df['image'].tolist()
        self.transform = transform

    def __getitem__(self, index):
        path, single_image = self.get_x(index, with_metadata=self.with_metadata)
        gts = self.get_y(index)
        if self.with_metadata:
            single_image, metadata = single_image
        single_image = self.transform(single_image)

        if self.with_metadata:
            return path, single_image, metadata, gts
        else:
            return path, single_image, gts

    def get_x(self, idx, with_metadata):

        img = Image.open(self.image_path[idx])
        # TODO: Properly load metadata here
        if with_metadata:
            pass
        return self.image_path[idx], img

    def get_y(self, idx):
        return torch.IntTensor(self.df.iloc[idx, 1:-1].tolist())

    def __len__(self):
        return len(self.image_path)


class ISIC19TestSet(data.Dataset):

    def __init__(self, csvfile, transform):

        self.df = pd.read_csv(csvfile)
        self.csv_columns = list(self.df.columns.values)[:-1]
        self.image_path = self.df['image'].apply(lambda x: x+'.jpg')
        self.transform = transform

    def __getitem__(self, index):
        path = self.get_x(index)
        single_image = self.get_y(index)
        single_image = self.transform(single_image)
        return path, single_image

    def get_x(self, idx):
        return self.image_path[idx]

    def get_y(self, idx):
        img = Image.open(self.image_path[idx])
        return img

    def __len__(self):
        return len(self.image_path)


class CustomEnsembleDatasetIn(data.Dataset):

    def __init__(self, dataset, gts, remove_labels, keep_indices, transform=None):

        remove_label_indices = [i for i, x in enumerate(gts) if x in remove_labels]
        keep_indices = list(set(keep_indices) - set(remove_label_indices))

        self.keep_indices = keep_indices
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):

        x, y = self.dataset[index]
        if self.transform is None:
            return x, y
        else:
            return self.transform(x), y

    def __len__(self):
        return self.dataset.__len__()


class CustomEnsembleDatasetInGeneric(data.Dataset):

    def __init__(self, dataset, gts, remove_labels, keep_indices, transform=None):

        remove_label_indices = [i for i, x in enumerate(gts) if x in remove_labels]
        keep_indices = list(set(keep_indices) - set(remove_label_indices))

        self.keep_indices = keep_indices
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):

        x, y = self.dataset[index]
        if self.transform is None:
            return x, y
        else:
            return self.transform(x), y

    def __len__(self):
        return self.dataset.__len__()


class CustomEnsembleDatasetOut(data.Dataset):

    def __init__(self, dataset, gts, remove_labels, keep_indices, transform=None):

        remove_label_indices = [i for i, x in enumerate(gts) if x not in remove_labels]
        keep_indices = list(set(keep_indices) - set(remove_label_indices))

        self.keep_indices = keep_indices
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):

        x, y = self.dataset[index]
        if self.transform is None:
            return x, y
        else:
            return self.transform(x), y

    def __len__(self):
        return self.dataset.__len__()


class TinyImageNetDataset(data.Dataset):

    def __init__(self, root='./Datasetstiny-imagenet-200', train=True, transform=None):
        self.train = train
        if os.path.exists(root):
            self.root = root
        else:
            self.root = root.replace('storage', 'storage')
        self.transform = transform
        self.train_dir = os.path.join(self.root, "train")
        self.val_dir = os.path.join(self.root, "val")

        if self.train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.train)

        words_file = os.path.join(self.root, "words.txt")
        wnids_file = os.path.join(self.root, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3,5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir,d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def get_targets(self):
        return [target for (_, target) in self.images]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


class GenericImageFolderDataset(data.Dataset):

    def __init__(self, root, type='dogs', train=True, transform=None, subset_index=None):

        self.train = train
        self.root = root
        self.transform = transform
        self.type = type
        self.train_dir = os.path.join(self.root, "Train")
        self.val_dir = os.path.join(self.root, "Test")
        self.subset_index = subset_index

        if self.train:
            self._create_class_idx_dict(self.train_dir)
            self._make_dataset(self.train_dir)
        else:
            self._create_class_idx_dict(self.val_dir)
            self._make_dataset(self.val_dir)

    def _create_class_idx_dict(self, dir):
        num_images = 0
        for root, dirs, files in os.walk(dir):
            for f in files:
                if f.endswith(".jpg") or f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        if self.subset_index is None:
            if self.type == 'dogs':
                if os.path.exists('./Datasets'):
                    dic = {}
                    with open('./DatasetsDogs/Stanford/stanford_classes_dict.pickle', 'rb') as dic_pickle:
                        temp_dic = pickle.load(dic_pickle)
                        for key, value in temp_dic.items():
                            if self.train:
                                dic[os.path.join(self.root, "Train/", key)] = value
                            else:
                                dic[os.path.join(self.root, "Test/", key)] = value
            elif self.type == 'tiny':
                if os.path.exists('./Datasets'):
                    dic = {}
                    with open(os.path.join(self.root, 'classes_dict.pickle'), 'rb') as dic_pickle:
                        temp_dic = pickle.load(dic_pickle)
                        for key, value in temp_dic.items():
                            if self.train:
                                dic[os.path.join(self.root, "Train/", key)] = value
                            else:
                                dic[os.path.join(self.root, "Test/", key)] = value
                else:
                    dic = {}
                    with open('./DatasetsDogs/Stanford/stanford_classes_dict.pickle', 'rb') as dic_pickle:
                        temp_dic = pickle.load(dic_pickle)
                        for key, value in temp_dic.items():
                            if self.train:
                                dic[os.path.join(self.root, "Train/", key)] = value
                            else:
                                dic[os.path.join(self.root, "Test/", key)] = value
            else:
                if os.path.exists('./Datasets'):
                    dic = {}
                    with open('./DatasetsBirds/nabirds/birdsdict.pickle', 'rb') as dic_pickle:
                        temp_dic = pickle.load(dic_pickle)
                        for key, value in temp_dic.items():
                            if self.train:
                                dic[os.path.join(self.root, "Train/", key)] = value
                            else:
                                dic[os.path.join(self.root, "Test/", key)] = value
                else:
                    dic = {}
                    with open('./DatasetsBirds/nabirds/birdsdict.pickle', 'rb') as dic_pickle:
                        temp_dic = pickle.load(dic_pickle)
                        for key, value in temp_dic.items():
                            if self.train:
                                dic[os.path.join(self.root, "Train/", key)] = value
                            else:
                                dic[os.path.join(self.root, "Test/", key)] = value
        else:
            if self.type == 'dogs':
                if os.path.exists('./Datasets'):
                    dic = {}
                    with open(f'./DatasetsDogs/Stanford/stanford_classes_dict_subset_{self.subset_index}.pickle', 'rb') as dic_pickle:
                        temp_dic = pickle.load(dic_pickle)
                        for key, value in temp_dic.items():
                            if self.train:
                                dic[os.path.join(self.root, "Train/", value)] = key
                            else:
                                dic[os.path.join(self.root, "Test/", value)] = key
                else:
                    dic = {}
                    with open(f'./DatasetsDogs/Stanford/stanford_classes_dict_subset_{self.subset_index}.pickle', 'rb') as dic_pickle:
                        temp_dic = pickle.load(dic_pickle)
                        for key, value in temp_dic.items():
                            if self.train:
                                dic[os.path.join(self.root, "Train/", value)] = key
                            else:
                                dic[os.path.join(self.root, "Test/", value)] = key
            else:
                if os.path.exists('./Datasets'):
                    dic = {}
                    with open(f'./DatasetsBirds/nabirds/nabirds_dict_subset_{self.subset_index}.pickle', 'rb') as dic_pickle:
                        temp_dic = pickle.load(dic_pickle)
                        for key, value in temp_dic.items():
                            if self.train:
                                dic[os.path.join(self.root, "Train/", key)] = value
                            else:
                                dic[os.path.join(self.root, "Test/", key)] = value
                else:
                    dic = {}
                    with open(f'./DatasetsBirds/nabirds/nabirds_dict_subset_{self.subset_index}.pickle', 'rb') as dic_pickle:
                        temp_dic = pickle.load(dic_pickle)
                        for key, value in temp_dic.items():
                            if self.train:
                                dic[os.path.join(self.root, "Train/", key)] = value
                            else:
                                dic[os.path.join(self.root, "Test/", key)] = value
        self.tgt_idx_to_class = {v: k for k, v in dic.items()}
        self.class_to_tgt_idx = dic

        self.num_classes = len(self.tgt_idx_to_class.keys())

    def _make_dataset(self, dir):
        self.images = []
        img_root_dir = dir
        list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".jpg") or fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_tgt_idx[tgt])
                        self.images.append(item)

    def get_targets(self):
        return [target for (_, target) in self.images]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        try:
            img_path, tgt = self.images[idx]
        except IndexError:
            print(idx)
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            if self.type != 'tiny':
                sample = self.transform[0](sample)
            else:
                sample = self.transform(sample)
        return sample, tgt


class MyDatasetFolder(torchvision.datasets.folder.VisionDataset):

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(MyDatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = torchvision.datasets.folder.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class MyImageFolder(MyDatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=torchvision.datasets.folder.default_loader, is_valid_file=None):
        super(MyImageFolder, self).__init__(root, loader, torchvision.datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
