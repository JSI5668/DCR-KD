import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class Minicity(data.Dataset):
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
        CamvidClass('Road', 0, 0, 'Road', 0, False, False, (128, 64, 128)),
        CamvidClass('sidewalk', 1, 1, 'sidewalk', 1, False, False, (244, 35, 232)),
        CamvidClass('building', 2, 2, 'building', 2, False, False, (70, 70, 70)),
        CamvidClass('wall', 3, 3, 'wall', 3, False, False, (102, 102, 156)),
        CamvidClass('fence', 4, 4, 'fence', 4, False, False, (190, 153, 153)),
        CamvidClass('pole', 5, 5, 'pole', 5, False, False, (153, 153, 153)),
        CamvidClass('traffic light', 6, 6, 'traffic light', 6, False, False, (250, 170, 30)),
        CamvidClass('traffic sign', 7, 7, 'traffic sign', 7, False, False, (220, 220, 0)),
        CamvidClass('vegetation', 8, 8, 'vegetation', 8, False, False, (107, 142, 35)),
        CamvidClass('terrain', 9, 9, 'terrain', 9, False, False, (152, 251, 152)),
        CamvidClass('sky', 10, 10, 'sky', 10, False, False, (70, 130, 180)),
        CamvidClass('person', 11, 11, 'person', 11, False, False, (220, 20, 60)),
        CamvidClass('rider', 12, 12, 'rider', 12, False, False, (255, 0, 0)),
        CamvidClass('car', 13, 13, 'car', 13, False, False, (0, 0, 142)),
        CamvidClass('truck', 14, 14, 'truck', 14, False, False, (0, 0, 70)),
        CamvidClass('bus', 15, 15, 'bus', 15, False, False, (0, 60, 100)),
        CamvidClass('train', 16, 16, 'train', 16, False, False, (0, 80, 100)),
        CamvidClass('motorcycle', 17, 17, 'motorcycle', 17, False, False, (0, 0, 230)),
        CamvidClass('bicycle', 18, 18, 'bicycle', 18, False, False, (119, 11, 32)),
        CamvidClass('unlabelled', -1, 255, 'void', 19, False, True, (0, 0, 0)),
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
        target[target == 255] = 19
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