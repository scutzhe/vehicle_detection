#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : demo.py
# @time    : 10/15/20 8:43 AM
# @desc    : 
'''

import os
import sys
import cv2
import argparse
from vision.ssd.config.fd_config import define_img_size
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

parser = argparse.ArgumentParser(description='detect_imgs')
parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=384, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.6, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1500, type=int,
                    help='nms candidate size')
parser.add_argument('--image_dir', default="vehicle_images", type=str,
                    help='imgs dir')
parser.add_argument("--image_path",default="01.jpg", type=str,
                    help="image_path")
parser.add_argument("--result_dir",default="/home/zhex/test_result/vehicle_detection", type=str,
                    help="result_dir")
parser.add_argument("--label_path",default="models_vehicle/labels.txt",type=str,
                    help="label_path")
parser.add_argument('--test_device', default="cuda:0", type=str,
                    help='cuda:0 or cpu')
args = parser.parse_args()

def detection():
    define_img_size(args.input_size)
    test_device = args.test_device
    class_names = [name.strip() for name in open(args.label_path).readlines()]

    if args.net_type == 'slim':
        model_path = ""
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
    elif args.net_type == 'RFB':
        model_path = "models_vehicle/RFB-Epoch-20-Loss-2.2841168881838954.pth"
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
    else:
        print("net type is wrong!")
        sys.exit(1)
    net.load(model_path)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    sum = 0
    for name in os.listdir(args.image_dir):
        img_path = os.path.join(args.image_dir, name)
        orig_image = cv2.imread(img_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(image, args.candidate_size / 2, args.threshold)
        sum += boxes.size(0)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{probs[i]:.2f}"
            # cv2.putText(orig_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(orig_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(args.result_dir, name), orig_image)
        print(f"Found {len(probs)} vehicle. The output image is {args.result_dir}")
    print(sum)

if __name__ == '__main__':
    detection()