#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  :
# @license :
# @contact : dylenzheng@gmail.com
# @file    : predeal.py
# @time    : 10/13/20 8:07 PM
# @desc    : 
'''
import os
import cv2
import random
import shutil
import numpy as np
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

def quantization_images(origin_image_dir,quantization_image_dir):
    """
    @param origin_image_dir:
    @param quantization_image_dir:
    @return:
    """
    assert os.path.exists(origin_image_dir),"{} is null".format(origin_image_dir)
    if not os.path.exists(quantization_image_dir):
        os.makedirs(quantization_image_dir)
    index = 0
    for name in tqdm(os.listdir(origin_image_dir)):
        image_path = os.path.join(origin_image_dir,name)
        shutil.copy(image_path,quantization_image_dir)
        index += 1
        if index == 500:
            break

def deal_quantization_images(origin_image_dir,save_image_dir):
    """
    @param origin_image_dir:
    @param save_image_dir:
    @return:
    """
    assert os.path.exists(origin_image_dir),"{} is null".format(origin_image_dir)
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
    for name in tqdm(os.listdir(origin_image_dir)):
        image_path = os.path.join(origin_image_dir,name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(384,192),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(save_image_dir+"/"+name,image)

#计算GPU使用率
def gpu_usage_rate(txt_path):
    """
    @param txt_path:
    @return:
    """
    assert os.path.exists(txt_path),"{} is null".format(txt_path)
    file_txt = open(txt_path,"r")
    sum = 0
    index = 0
    for line in tqdm(file_txt.readlines()):
        num = line.strip().split("@")[0]
        sum += int(num)
        index += 1
    print("usage_rate:",round(sum/index,4),index)

#计算CPU使用率
def cpu_usage_rate(txt_path_one,txt_path_two):
    """
    @param txt_path_one:
    @param txt_path_two:
    @return:
    """
    assert os.path.exists(txt_path_one), "{} is null".format(txt_path_one)
    assert os.path.exists(txt_path_two), "{} is null".format(txt_path_two)
    file_txt_one = open(txt_path_one, "r")
    file_txt_two = open(txt_path_two, "r")
    info_one = file_txt_one.readlines()
    info_two = file_txt_two.readlines()
    s0 = 0
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0

    r02 = 0
    r12 = 0
    r22 = 0
    r32 = 0
    r42 = 0
    r52 = 0

    s01 = 0
    s11 = 0
    s21 = 0
    s31 = 0
    s41 = 0
    s51 = 0

    r021 = 0
    r121 = 0
    r221 = 0
    r321 = 0
    r421 = 0
    r521 = 0



    for line in info_one:
        flag = line.strip().split(" ")
        if flag[0] == "cpu0":
            r02 = int(flag[4])
            for item in flag[1:]:
                s0 += int(item)
        if flag[0] == "cpu1":
            r12 = int(flag[4])
            for item in flag[1:]:
                s1 += int(item)
        if flag[0] == "cpu2":
            r22 = int(flag[4])
            for item in flag[1:]:
                s2 += int(item)
        if flag[0] == "cpu3":
            r32 = int(flag[4])
            for item in flag[1:]:
                s3 += int(item)
        if flag[0] == "cpu4":
            r42 = int(flag[4])
            for item in flag[1:]:
                s4 += int(item)
        if flag[0] == "cpu5":
            r52 = int(flag[4])
            for item in flag[1:]:
                s5 += int(item)

    for line in info_two:
        flag = line.strip().split(" ")
        if flag[0] == "cpu0":
            r021 = int(flag[4])
            for item in flag[1:]:
                s01 += int(item)
        if flag[0] == "cpu1":
            r121 = int(flag[4])
            for item in flag[1:]:
                s11 += int(item)
        if flag[0] == "cpu2":
            r221 = int(flag[4])
            for item in flag[1:]:
                s21 += int(item)
        if flag[0] == "cpu3":
            r321 = int(flag[4])
            for item in flag[1:]:
                s31 += int(item)
        if flag[0] == "cpu4":
            r421 = int(flag[4])
            for item in flag[1:]:
                s41 += int(item)
        if flag[0] == "cpu5":
            r521 = int(flag[4])
            for item in flag[1:]:
                s51 += int(item)
    # print("s10,s11,s12,s13,s14,s15:",s0,s1,s2,s3,s4,s5)
    # print("s20,s21,s22,s23,s24,s25:",s01,s11,s21,s31,s41,s51)
    # print("s20-s10,s21-s11,s22-s12,s23-s13,s24-s4,s25-s15:",
    #                 s01-s0,s11-s1,s21-s2,s31-s3,s41-s4,s51-s5)
    print("rate0,rate1,rate2,rate3,rate4,rate5:",
          ((s01-s0) - (r021 - r02)) / (s01-s0),
          ((s11-s1) - (r121 - r12)) / (s11-s1),
          ((s21-s2) - (r221 - r22)) / (s21-s2),
          ((s31-s3) - (r321 - r32)) / (s31-s3),
          ((s41-s4) - (r421 - r42)) / (s41-s4),
          ((s51-s5) - (r521 - r52)) / (s51-s5)
          )

def detection_cost(txt_path):
    """
    @param txt_path:
    @return:
    """
    assert os.path.exists(txt_path),"{} is null".format(txt_path)
    filex = open(txt_path,"r")
    sum = 0
    index = 0
    for line in filex.readlines():
        line = line.strip().split("m")[0]
        sum += int(line)
        index += 1
    print("avg:",round(sum/index,3))

def split_drink(labels_dir,images_dir):
    """
    @param labels_dir:
    @param images_dir:
    @return:
    """
    assert os.path.exists(labels_dir),"{} is null".format(labels_dir)
    assert os.path.exists(images_dir),"{} is null".format(images_dir)
    file_one = open("annotation.txt","a")
    for name in tqdm(os.listdir(labels_dir)):
        label_path = os.path.join(labels_dir,name)
        image_path = os.path.join(images_dir,name.split(".")[0]+".jpg")
        file_one.write(name.split(".")[0] + ".jpg")
        H,W = cv2.imread(image_path).shape[:2]
        filex = open(label_path,"r")
        for line in filex.readlines():
            info = line.strip().split(" ")[1:]
            x_min = round((float(info[0]) - float(info[2])/2) * W)
            y_min = round((float(info[1]) - float(info[3])/2) * H)
            x_max = round((float(info[0]) + float(info[2])/2) * W)
            y_max = round((float(info[1]) + float(info[3])/2) * H)
            file_one.write(" " + str(x_min) + " " + str(y_min) + " " +str(x_max) + " " + str(y_max))
        file_one.write("\n")

def check_drink(annotation_path,images_dir):
    """
    @param annotation_path:
    @param images_dir:
    @return:
    """
    assert  os.path.exists(annotation_path),"{} is null".format(annotation_path)
    assert  os.path.exists(images_dir),"{} is null".format(images_dir)
    filex = open(annotation_path,"r")
    for line in tqdm(filex.readlines()):
        info = line.strip().split(" ")
        image_path = os.path.join(images_dir,info[0])
        image = cv2.imread(image_path)
        x_min = int(info[1])
        y_min = int(info[2])
        x_max = int(info[3])
        y_max = int(info[4])
        cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(0,0,255),2)
        cv2.imshow("image",image)
        cv2.waitKey(1000)

def split_drink_train_val(annotation_path):
    """
    @param annotation_path:
    @return:
    """
    assert os.path.exists(annotation_path),"{} is null".format(annotation_path)
    train_file = open("train.txt","a")
    val_file = open("val.txt","a")
    info = open(annotation_path,"r").readlines()
    length = len(info)
    train_subset = random.sample(info,int(0.8*length))
    val_subset = set(info).difference(train_subset)
    for line in tqdm(train_subset):
        train_file.write(line)
    for line in tqdm(val_subset):
        val_file.write(line)




# if __name__ == '__main__':
#     ori_image_dir = "/home/zhex/data/vehicle/JPEGImages"
#     train_file_path ="/home/zhex/data/vehicle/train.txt"
#     val_file_path ="/home/zhex/data/vehicle/val.txt"
#     train_file = open(train_file_path,"a")
#     val_file = open(val_file_path,"a")
#     split_train_val(ori_image_dir,train_file,val_file)
#     annotation_dir = "/home/zhex/data/vehicle/Annotations"
    # image_dir = "/home/zhex/data/vehicle/JPEGImages"
    # check(annotation_dir, image_dir)
    # delete_empty_annotation(annotation_dir,image_dir)


# if __name__ == "__main__":
    # origin_image_dir = "/home/zhex/data/vehicle/JPEGImages"
    # quantization_image_dir = "/home/zhex/data/quantization_images"
    # quantization_save_image_dir = "/home/zhex/data/quantization_images_new"
    # quantization_images(origin_image_dir,quantization_image_dir)
    # deal_quantization_images(quantization_image_dir,quantization_save_image_dir)
    # txt_path = "fre.txt"
    # usage_rate(txt_path)
    # cpu_usage_rate(txt_path1,txt_path2)
    # txt_path = "time_cost.txt"
    # detection_cost(txt_path)
    # gpu_usage_rate(txt_path)

if __name__ == "__main__":
    lables_dir = "/home/zhex/data/drink/labels"
    images_dir = "/home/zhex/data/drink/images"
    # split_drink(lables_dir,images_dir)
    annotation_path = "annotation.txt"
    # check_drink(annotation_path,images_dir)
    split_drink_train_val(annotation_path)