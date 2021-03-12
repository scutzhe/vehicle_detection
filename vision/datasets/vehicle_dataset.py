#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  :
# @license :
# @contact : dylenzheng@gmail.com
# @file    : vehicle_dataset.py
# @time    : 10/13/20 7:33 PM
# @desc    : 
'''
import os
import cv2
import logging
import numpy as np
import xml.etree.ElementTree as ET

class VehicleDataset:
    def __init__(self, root, is_train = False, transform=None, target_transform=None):
        self.root = root
        self.image_dir = self.root + "/" + "JPEGImages"
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        if is_train:
                image_sets_file = self.root + "/" + "train.txt"
        else:
                image_sets_file = self.root + "/" + "val.txt"
        self.ids = VehicleDataset._read_image_ids(image_sets_file)

        label_file_name = self.root + "/" + "labels.txt"
        if os.path.isfile(label_file_name):
            with open(label_file_name, 'r') as infile:
                classes = infile.read().splitlines()

            classes.insert(0, 'BACKGROUND')
            self.class_names = tuple(classes)
            logging.info("vehicle Labels read from file: " + str(self.class_names))
        else:
            logging.info("No labels file, using default vehicle classes.")
            self.class_names = ("BACKGROUND", "vehicle")
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        # print("image.size(),boxes.size(),labels.size():",image.size(),boxes.size(),labels.size())
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_path = self.root + "/" + "Annotations/{}.xml".format(image_id)
        # print("annotation_path:",annotation_path)
        # data = ET.parse(annotation_path)解析xml文件
        # data.findall("object")查找data名下所有名为object的元素
        objects = ET.parse(annotation_path).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            if class_name in self.class_dict:
                bbox = object.find("bndbox")
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_path = self.root  + "/" +"JPEGImages/{}.jpg".format(image_id)
        # print("image_path:",image_path)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
