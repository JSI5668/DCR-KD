import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

## 분할할 class 제외하고 나머지 다 Backgound 로 !!

class Camvid_proposed(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CamvidClass = namedtuple('CamvidClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CamvidClass('sky', -1, 255, 'sky', 0, False, True, (0, 0, 0)),
        CamvidClass('building', 0, 0, 'building', 1, False, False, (128, 128, 128)),
        CamvidClass('pole', 1, 1, 'pole', 2, False, False, (128, 0, 0)),
        CamvidClass('road', 0, 0, 'road', 3, False, False, (128, 128, 128)),
        CamvidClass('sidewalk', 1, 1, 'sidewalk', 4, False, False, (128, 0, 0)),
        CamvidClass('tree', 1, 1, 'tree', 5, False, False, (128, 0, 0)),
        CamvidClass('signsymbol', 1, 1, 'signsymbol', 6, False, False, (128, 0, 0)),
        CamvidClass('fence', 1, 1, 'fence', 7, False, False, (128, 0, 0)),
        CamvidClass('car', 1, 1, 'car', 8, False, False, (128, 0, 0)),
        CamvidClass('pedestrian', 1, 1, 'pedestrian', 9, False, False, (128, 0, 0)),
        CamvidClass('bicyclist', 1, 1, 'bicyclist', 10, False, False, (128, 0, 0)),
        CamvidClass('unlabelled', -1, 255, 'void', 11, False, True, (0, 0, 0)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    # train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    # train_id_to_color = np.array(train_id_to_color)
    # id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):  #추가 1: root_2
        self.root = os.path.expanduser(root)
        # self.root_2 = os.path.expanduser(root_2) #추가 2
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root + '/', 'leftImg8bit/', split)
        # self.images_dir_2 = os.path.join(self.root_2 + '/', 'leftImg8bit/', split)  ### 추가 3

        self.targets_dir = os.path.join(self.root + '/', self.mode + '/', split)
        # self.targets_dir_2 = os.path.join(self.root_2 + '/', self.mode + '/', split) ## 추가 4
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        # self.images_2 = [] ## 추가 5
        # self.targets_2 = [] ##추가 6

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir + '/', city)
            target_dir = os.path.join(self.targets_dir + '/', city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir + '/', file_name))
                # target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                #                              self._get_target_suffix(self.mode, self.target_type))
                # self.targets.append(os.path.join(target_dir + '/', target_name))
                self.targets.append(os.path.join(target_dir + '/', file_name))
        #
        # for city in os.listdir(self.images_dir_2):   ## 추가 7
        #     img_dir_2 = os.path.join(self.images_dir_2 + '/', city)
        #     target_dir_2 = os.path.join(self.targets_dir_2 + '/', city)
        #
        #     for file_name in os.listdir(img_dir_2):
        #         self.images_2.append(os.path.join(img_dir_2 + '/', file_name))
        #         # target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
        #         #                              self._get_target_suffix(self.mode, self.target_type))
        #         # self.targets.append(os.path.join(target_dir + '/', target_name))
        #         self.targets_2.append(os.path.join(target_dir_2 + '/', file_name))


    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        # target[target == 255] = 19
        target[target == 255] = 11
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        # image_2 = Image.open(self.images_2[index]).convert('RGB') #추가 8
        # target_2 = Image.open(self.targets_2[index])

        if self.transform:
            image, target = self.transform(image, target)
            # image_2, target_2 = self.transform(image_2, target_2) #추가 9

        # image = torch.cat((image, image_2), dim = 0) #추가 10

        target = self.encode_target(target)
        # target_2 = self.encode_target(target_2) #추가 11

        # target = torch.cat((target, target_2), dim=0) #추가 12

        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)