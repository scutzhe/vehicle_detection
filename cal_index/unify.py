#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020
# @contact : dylenzheng@gmail.com
# @file    : unify.py
# @time    : 10/28/20 11:12 AM
# @desc    : 
'''
import os
from tqdm import tqdm
def common_difference(ori_gt_txt_path,ori_pre_txt_path,post_gt_txt_path,post_pre_txt_path):
    """
    @param ori_gt_txt_path:
    @param ori_pre_txt_path:
    @param post_gt_txt_path:
    @param post_pre_txt_path:
    @return:
    """
    assert os.path.exists(ori_gt_txt_path),"{} is null".format(ori_gt_txt_path)
    assert os.path.exists(ori_pre_txt_path),"{} is null".format(ori_pre_txt_path)
    gt_file = open(ori_gt_txt_path,"r")
    pre_file = open(ori_pre_txt_path,"r")
    gt_names = []
    pre_names = []
    for line in tqdm(gt_file.readlines()):
        name = line.strip().split(" ")[0]
        gt_names.append(name)
    for line in tqdm(pre_file.readlines()):
        name = line.strip().split(" ")[0]
        pre_names.append(name)
    common_names = [item for item in pre_names if item in gt_names]
    print("len(gt_names):",len(gt_names))
    print("len(pre_names):",len(pre_names))
    print("len(common_names):",len(common_names))
    difference_names = set(gt_names).difference(set(pre_names))
    gt_file.close()
    pre_file.close()


    gt_file = open(ori_gt_txt_path, "r")
    pre_file = open(ori_pre_txt_path, "r")
    post_gt_file = open(post_gt_txt_path,"a")
    post_pre_file = open(post_pre_txt_path,"a")
    for line in  tqdm(gt_file.readlines()):
        flag = line.strip().split(" ")[0]
        if flag in common_names:
            post_gt_file.write(line)
        else:
            pass

    for line in  tqdm(pre_file.readlines()):
        flag = line.strip().split(" ")[0]
        if flag in common_names:
            post_pre_file.write(line)
        else:
            pass

# if __name__ == '__main__':
#     ori_gt_txt_path = "gt.txt"
#     ori_pre_txt_path = "pre.txt"
#     post_gt_txt_path = "gt_post.txt"
#     post_pre_txt_path = "pre_post.txt"
#     common_difference(ori_gt_txt_path,ori_pre_txt_path,post_gt_txt_path,post_pre_txt_path)


