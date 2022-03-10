# -*- coding: utf-8 -*-
# @Date    : 10/6/19
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from fileinput import filename
from functools import partial
import sys
from termios import VMIN
import torch
import os
import PIL
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    download_file_from_google_drive,
    check_integrity,
    verify_str_arg,
)
from torch.utils.data import Dataset
import glob
import pandas as pd
from sdgym.datasets import load_dataset
import numpy


from sdgym.datasets import load_tables


class CelebA(Dataset):
    """pyTorch Dataset wrapper for the generic flat directory images dataset"""

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """
        file_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)

        # return the files list
        return files

    def __init__(self, root, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = root
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(img_name)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # return the image:
        return img, img


class FFHQ(Dataset):
    """pyTorch Dataset wrapper for the generic flat directory images dataset"""

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """
        file_names = (
            glob.glob(os.path.join(self.data_dir, "./*/*.png"))
            + glob.glob(os.path.join(self.data_dir, "./*.jpg"))
            + [
                y
                for x in os.walk(self.data_dir)
                for y in glob.glob(os.path.join(x[0], "*.webp"))
            ]
        )
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)

        # return the files list
        return files

    def __init__(self, root, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = root
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(img_name)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # return the image:
        return img, img


class Table(Dataset):
    """pyTorch Dataset wrapper for the table dataset"""

    def _setup_data(self):
        metadata = load_dataset("asia")
        tables = load_tables(metadata)

        data = tables["asia"]
        data = data.applymap(lambda x: {"no": False, "yes": True}[x]).to_numpy(
            numpy.uint8
        )
        # b = (numpy.arange(data.max()) == data[...,None]-1).astype(int)
        self.data = data

    def __init__(self, root, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_file = root
        self.transform = transform
        self._setup_data()
        # setup the files for reading
        # self.files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """

        row = self.data[idx]

        return row, row


class AdultTable(Dataset):
    """pyTorch Dataset wrapper for the table dataset"""

    def _setup_data(self):
        metadata = load_dataset("adult")
        tables = load_tables(metadata)

        data = tables["adult"]
        # data = data[~(data == "?").any(axis=1)]
        for col in data.columns:
            for i in range(data[col].size):
                if data[col].at[i] == "?":
                    data[col].at[i] = "?" + col

        self.num_size = 0
        self.cat_size = 0
        self.cats = []
        mapping = {}
        types = {"numerical": {}, "categorical": []}
        self.mapping = {}

        for field_name, field_type in metadata.get_fields("adult").items():
            if field_type["type"] == "categorical":
                self.cat_size += 1
                m = {x: i for i, x in enumerate(data[field_name].unique())}
                mapping = {**mapping, **m}

                self.cats.append(len(data[field_name].unique()))
                types[field_type["type"]].append(field_name)

                self.mapping[field_name] = {
                    key: value + sum(self.cats[:-1]) for key, value in m.items()
                }

            elif field_type["type"] == "numerical":
                self.num_size += 1

                types[field_type["type"]][field_name] = field_type["subtype"]

        # re_col_order = types["numerical"] + types["categorical"]
        # self.new_header = re_col_order
        self.types = types
        self.old_header = list(data.columns)

        num_data = data[types["numerical"].keys()]
        self.num_min_max, num_data = AdultTable.min_max_norm(num_data)

        data = pd.concat([num_data, data[types["categorical"]]], axis=1)
        self.new_header = data.columns

        self.data = data.replace(mapping).reset_index(drop=True).to_numpy()

    def __init__(self, root, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_file = root
        self.transform = transform
        self._setup_data()
        # setup the files for reading
        # self.files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        # print(self.data)
        # print(idx)
        row = self.data[idx]

        return row, row

    def get_cat_num_size(self):
        return self.cat_size, self.cats, self.num_size

    def min_max_norm(data: pd.DataFrame):
        data_dict = {}
        min_max_dict = {}
        for col in data.columns:
            a = data[col]
            v_min, v_max = a.min(), a.max()
            new_min, new_max = -1, 1
            a = (a - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
            data_dict[col] = a
            min_max_dict[col] = {"min": v_min, "max": v_max}
        return pd.DataFrame(min_max_dict), pd.DataFrame(data_dict)

    def recover_norm_data(self, data: pd.DataFrame):
        print(self.new_header)
        data = pd.DataFrame(data.values, columns=self.new_header)
        data_dict = {}
        print(sum(self.cats))
        for col in data.columns:
            if col in self.types["numerical"].keys():
                a = data[col]
                v_min, v_max = (
                    self.num_min_max[col]["min"],
                    self.num_min_max[col]["max"],
                )
                new_min, new_max = -1, 1
                a = (a - new_min) / (new_max - new_min) * (v_max - v_min) + v_min
                if self.types["numerical"][col] == "integer":
                    a = a.astype(int)
                data_dict[col] = a
            elif col in self.types["categorical"]:

                a = [
                    next(key for key, value in self.mapping[col].items() if value == x)
                    for x in data[col]
                ]

                a = ["?" if t == "?" + col else t for t in a]

                data_dict[col] = pd.Series(a)

        return pd.DataFrame(data_dict)
