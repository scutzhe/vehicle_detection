#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : convert_to_onnx.py
# @time    : 10/14/20 9:07 AM
# @desc    :
'''

import sys
import torch.onnx
from vision.ssd.config.fd_config import define_img_size
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd

input_img_size = 384  # define input size ,default optional(128/160/320/384/480/640/1280)
define_img_size(input_img_size)

# net_type = "slim"  # inference faster,lower precision
net_type = "RFB"     # inference lower,higher precision
label_path = "models_vehicle/labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

if net_type == 'slim':
    model_path = ""
    net = create_mb_tiny_fd(len(class_names), is_test=True)
elif net_type == 'RFB':
    model_path = "models_vehicle/RFB-Epoch-20-Loss-2.2841168881838954.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True)
else:
    print("No network type.")
    sys.exit(1)

net.load(model_path)
net.eval()
net.to("cuda")

model_path = f"models_vehicle/onnx/vehicle_detection_epoch_20.onnx"
# model_path = f"models_vehicle/onnx2mnn/vehicle_detection_epoch_20.onnx"

dummy_input = torch.randn(1, 3, 192, 384).to("cuda")
torch.onnx.export(net, dummy_input, model_path, verbose=False, input_names=['input'], output_names=['scores', 'boxes'])