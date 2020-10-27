#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020
# @contact : dylenzheng@gmail.com
# @file    : drink_dataset.py
# @time    : 10/27/20 3:20 PM
# @desc    : 
'''

import os
import cv2
import logging
import numpy as np

class DrinkDataset:
    def __init__(self, root, is_train = False, transform=None, target_transform=None):
        self.root = root
        self.image_dir = self.root + "/" + "images"
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        if is_train:
                image_sets_file_txt_path = self.root + "/" + "train.txt"
        else:
                image_sets_file_txt_path = self.root + "/" + "val.txt"
        self.info = DrinkDataset._read(image_sets_file_txt_path)

        label_file_name = self.root + "/" + "labels.txt"
        if os.path.isfile(label_file_name):
            with open(label_file_name, 'r') as infile:
                classes = infile.read().splitlines()

            classes.insert(0, 'BACKGROUND')
            self.class_names = tuple(classes)
            logging.info("vehicle Labels read from file: " + str(self.class_names))
        else:
            logging.info("No labels file, using default vehicle classes.")
            self.class_names = ("BACKGROUND", "drink")
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        id_coordination = self.info[index]
        image_path = os.path.join(self.image_dir,str(id_coordination[0]))
        coordination = id_coordination[1:]
        boxes, labels = self._get_annotation(coordination)
        image = self._read_image(image_path)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        # print("image.size(),boxes.size(),labels.size():",image.size(),boxes.size(),labels.size())
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.info[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.info[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.info)

    @staticmethod
    def _read(image_sets_path):
        info = []
        filex = open(image_sets_path,"r")
        for line in filex.readlines():
            tmp = line.strip().split(" ")
            info.append(tmp)
        return info

    def _get_annotation(self, coordination):
        """
        @param coordination:
        @return:
        """
        boxes = []
        labels = []
        for index in range(len(coordination)//4):
            x1 = float(coordination[index * 4])
            y1 = float(coordination[index * 4 + 1])
            x2 = float(coordination[index * 4 + 2])
            y2 = float(coordination[index * 4 + 3])
            boxes.append([x1,y1,x2,y2])
            labels.append(1)
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def _read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image