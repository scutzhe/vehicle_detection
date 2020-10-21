#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : mnn_inference.py
# @time    : 10/15/20 2:02 PM
# @desc    : 
'''
import os
import argparse
import time
import MNN
import cv2
import numpy as np
import torch

import vision.utils.box_utils_numpy as box_utils

parser = argparse.ArgumentParser(description='run vehicle detection with MNN in py')
parser.add_argument('--model_path', default="models_vehicle/mnn/vehicle_detection_epoch_20_simple.mnn", type=str, help='model path')
parser.add_argument('--input_size', default=384, type=int)
parser.add_argument('--threshold', default=0.7, type=float, help='score threshold')
parser.add_argument('--imgs_path', default="vehicle_images", type=str, help='imgs dir')
parser.add_argument('--results_path', default="results", type=str, help='results dir')
args = parser.parse_args()

image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
strides = [8, 16, 32, 64]

def define_img_size(size):
    img_size_dict = {128: [128, 96],
                     160: [160, 120],
                     320: [320, 240],
                     384: [384, 192],
                     480: [480, 360],
                     640: [640, 480],
                     1280: [1280, 960]}
    image_size = img_size_dict[size]

    feature_map_w_h_list_dict = {128: [[16, 8, 4, 2], [12, 6, 3, 2]],
                                 160: [[20, 10, 5, 3], [15, 8, 4, 2]],
                                 320: [[40, 20, 10, 5], [30, 15, 8, 4]],
                                 384: [[48, 24, 12, 6], [24, 12, 6, 3]],
                                 480: [[60, 30, 15, 8], [45, 23, 12, 6]],
                                 640: [[80, 40, 20, 10], [60, 30, 15, 8]],
                                 1280: [[160, 80, 40, 20], [120, 60, 30, 15]]}
    feature_map_w_h_list = feature_map_w_h_list_dict[size]

    shrinkage_list = []
    for i in range(0, len(image_size)):
        item_list = []
        for k in range(0, len(feature_map_w_h_list[i])):
            item_list.append(image_size[i] / feature_map_w_h_list[i][k])
        shrinkage_list.append(item_list)
    print("feature_map_w_h_list:",feature_map_w_h_list)
    print("shrinkage_list:",shrinkage_list)
    print("image_size:",image_size)
    print("min_boxes:",min_boxes)
    index = 0
    print("exec_index:",index)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    index += 1
    print("exec_index:",index)
    return priors

def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True) -> torch.Tensor:
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h
                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([x_center,y_center,w,h])

    print("priors nums:{}".format(len(priors)))
    index = 0
    for(x1,y1,x2,y2) in priors:
        print("priors_index_{}:".format(index),x1,y1,x2,y2)
        index += 1
        if index == 4:
            break
    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def inference():
    input_size = args.input_size
    priors = define_img_size(input_size)
    result_path = args.results_path
    imgs_path = args.imgs_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    listdir = os.listdir(imgs_path)
    for file_path in listdir:
        img_path = os.path.join(imgs_path, file_path)
        image_ori = cv2.imread(img_path)
        interpreter = MNN.Interpreter(args.model_path)
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)
        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (384,192))
        image = image.astype(float)
        image = (image - image_mean) / image_std
        image = image.transpose((2, 0, 1))
        tmp_input = MNN.Tensor((1, 3, 192, 384), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFrom(tmp_input)
        time_time = time.time()
        interpreter.runSession(session)
        scores = interpreter.getSessionOutput(session, "scores").getData()
        boxes = interpreter.getSessionOutput(session, "boxes").getData()
        # print("scores[0]:",scores[0])
        # print("scores[1]:",scores[1])
        # print("scores[2]:",scores[2])
        # print("scores[3]:",scores[3])
        # print("boxes[0]:",boxes[0])
        # print("boxes[1]:",boxes[1])
        # print("boxes[2]:",boxes[2])
        # print("boxes[3]:",boxes[3])
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        # print("boxes.shape:",boxes.shape)
        # print("scores.shape:",scores.shape)
        # print("priors.size:",priors.size())
        print("inference time: {} s".format(round(time.time() - time_time, 4)))
        boxes = box_utils.convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
        boxes = box_utils.center_form_to_corner_form(boxes)
        boxes, labels, probs = predict(image_ori.shape[1], image_ori.shape[0], scores, boxes, args.threshold)
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            cv2.rectangle(image_ori, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("UltraFace_mnn_py", image_ori)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()

def inference_image(image_path):
    input_size = args.input_size
    priors = define_img_size(input_size)
    print("priors:",priors[0][0],priors[0][1],priors[0][2],priors[0][3])
    image_ori = cv2.imread(image_path)
    interpreter = MNN.Interpreter(args.model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (384,192))
    image = image.astype(float)
    image = (image - image_mean) / image_std
    image = image.transpose((2, 0, 1))
    tmp_input = MNN.Tensor((1, 3, 192, 384), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    time_time = time.time()
    interpreter.runSession(session)
    scores = interpreter.getSessionOutput(session, "scores").getData()
    boxes = interpreter.getSessionOutput(session, "boxes").getData()
    # print("scores[0]:", scores[0])
    # print("scores[1]:", scores[1])
    # print("scores[2]:", scores[2])
    # print("scores[3]:", scores[3])
    # print("boxes[0]:", boxes[0])
    # print("boxes[1]:", boxes[1])
    # print("boxes[2]:", boxes[2])
    # print("boxes[3]:", boxes[3])
    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
    # print("boxes.shape:",boxes.shape)
    # print("scores.shape:",scores.shape)
    # print("priors.size:",priors.size())
    print("inference time: {} s".format(round(time.time() - time_time, 4)))
    boxes = box_utils.convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
    boxes = box_utils.center_form_to_corner_form(boxes)
    boxes, labels, probs = predict(image_ori.shape[1], image_ori.shape[0], scores, boxes, args.threshold)
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        cv2.rectangle(image_ori, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow("UltraFace_mnn_py", image_ori)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # inference()
    # image_path = "vehicle_images/01.jpg"
    image_path = "/home/zhex/test_result/vehicle_detection/03.jpeg"
    inference_image(image_path)
