import torch
from torch.utils import data
import numpy as np
from PIL import Image
import pandas as pd
import random
import ipdb


# global_seed = 1
# torch.backends.cudnn.deterministic = True
# random.seed(global_seed)
# np.random.seed(global_seed)
# torch.manual_seed(global_seed)
# torch.cuda.manual_seed(global_seed)


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

        try:
            img = Image.open(self.image_path[idx])
        except IndexError:
            print(idx)
            print(len(self.image_path))
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

        try:
            img = Image.open(self.image_path[idx])
        except IndexError:
            ipdb.set_trace()
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

        try:
            img = Image.open(self.image_path[idx])
        except IndexError:
            ipdb.set_trace()
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

        try:
            img = Image.open(self.image_path[idx])
        except IndexError:
            ipdb.set_trace()
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

    def __init__(self, dataset, remove_labels, keep_indices, transform=None):

        remove_label_indices = [i for i, x in enumerate(dataset.targets) if x in remove_labels]
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

    def __init__(self, dataset, remove_labels, keep_indices, transform=None):

        remove_label_indices = [i for i, x in enumerate(dataset.targets) if x not in remove_labels]
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
