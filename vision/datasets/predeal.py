#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : predeal.py
# @time    : 10/13/20 8:07 PM
# @desc    : 
'''
import os
import cv2
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm

def check(annotation_dir,image_dir):
    """
    @param annotation_dir:
    @param image_dir:
    @return:
    """
    assert os.path.exists(annotation_dir),"{} is null"
    assert os.path.exists(annotation_dir),"{} is null"
    save_dir = "result"
    for name in tqdm(os.listdir(annotation_dir)):
        image_path = os.path.join(image_dir,str(name.split(".")[0])+".jpg")
        image = cv2.imread(image_path)
        annotation_path = os.path.join(annotation_dir,name)
        objects = ET.parse(annotation_path).findall("object")
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            print("class_name:",class_name)
            bbox = object.find('bndbox')
            x1 = int(bbox.find('xmin').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),1)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_dir + "/"+str(name.split(".")[0])+".jpg",image)
        # cv2.imshow("image",image)
        # cv2.waitKey(10000)


def delete_empty_annotation(annotation_dir,image_dir):
    """
    @param annotation_dir:
    @param image_dir:
    @return:
    """
    import shutil
    assert os.path.exists(annotation_dir), "{} is null"
    assert os.path.exists(annotation_dir), "{} is null"
    no_annotation_dir = "/home/zhex/data/no_annotation_dir"
    no_image_dir = "/home/zhex/data/no_image_dir"
    if not os.path.exists(no_image_dir):
        os.makedirs(no_image_dir)
    if not os.path.exists(no_annotation_dir):
        os.makedirs(no_annotation_dir)
    num = 0
    for name in tqdm(os.listdir(annotation_dir)):
        image_path = os.path.join(image_dir, str(name.split(".")[0]) + ".jpg")
        annotation_path = os.path.join(annotation_dir, name)
        objects = ET.parse(annotation_path).findall("object")
        boxes = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # print("class_name:", class_name)
            bbox = object.find('bndbox')
            x1 = int(bbox.find('xmin').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
            boxes.append([x1,y1,x2,y2])
        if len(boxes)==0:
            shutil.move(annotation_path,no_annotation_dir)
            shutil.move(image_path,no_image_dir)
            num += 1
    print("empty annotation file num:",num)


def split_train_val(ori_image_dir,train_file,val_file):
    """
    @param ori_image_dir:
    @param train_file:
    @param val_file:
    @return:
    """
    assert os.path.exists(ori_image_dir),"{} is null !!!".format(ori_image_dir)
    txt_names = os.listdir(ori_image_dir)
    train_names = random.sample(txt_names,int(0.8*len(txt_names)))
    val_names = set(txt_names).difference(set(train_names))
    print("train_size:",len(train_names))
    print("val_size:",len(val_names))
    for name in tqdm(train_names):
        train_file.write(name.split(".")[0]+"\n")
    for name in tqdm(val_names):
        val_file.write(name.split(".")[0]+"\n")

if __name__ == '__main__':
    ori_image_dir = "/home/zhex/data/vehicle/JPEGImages"
    train_file_path ="/home/zhex/data/vehicle/train.txt"
    val_file_path ="/home/zhex/data/vehicle/val.txt"
    train_file = open(train_file_path,"a")
    val_file = open(val_file_path,"a")
    split_train_val(ori_image_dir,train_file,val_file)
#     annotation_dir = "/home/zhex/data/vehicle/Annotations"
    # image_dir = "/home/zhex/data/vehicle/JPEGImages"
    # check(annotation_dir, image_dir)
    # delete_empty_annotation(annotation_dir,image_dir)