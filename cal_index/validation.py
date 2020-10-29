#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020
# @contact : dylenzheng@gmail.com
# @file    : validation.py
# @time    : 10/28/20 11:30 AM
# @desc    : 
'''

def get_info(txt_file_path):
	"""
	@param txt_file_path:
	@return:
	"""
	with open(txt_file_path, 'r') as f:
		lines = f.readlines()
	bbox_num = 0
	info_dict = {}
	for line in lines:
		info = line.strip().split(" ")
		img_id = info[0]
		coord_score = info[1:]
		bbox_num += len(coord_score)//4
		info_dict[img_id] = list()

		for index in range(len(coord_score)//4):
			x1 = int(coord_score[index * 4])
			y1 = int(coord_score[index * 4 + 1])
			x2 = int(coord_score[index * 4 + 2])
			y2 = int(coord_score[index * 4 + 3])
			info_dict[img_id].append([x1,y1,x2,y2])

	return bbox_num,info_dict


def iou(predict_bbox, ground_truth_bbox):
	'''
	:param predict_bbox: list, format,[x1,y1,x2,y2,conf]
	:param ground_truth_bbox: list, format, [x1,y1,x2,y2,conf]
	:return: value,area of area
	'''
	predict_area = (predict_bbox[2] - predict_bbox[0])*(predict_bbox[3] - predict_bbox[1])
	ground_truth_area = (ground_truth_bbox[2] - ground_truth_bbox[0])*(ground_truth_bbox[3] - ground_truth_bbox[1])
	inter_x = min(predict_bbox[2],ground_truth_bbox[2]) - max(predict_bbox[0],ground_truth_bbox[0])
	inter_y = min(predict_bbox[3],ground_truth_bbox[3]) - max(predict_bbox[1],ground_truth_bbox[1])
	if inter_x<=0 or inter_y<=0:
		return 0
	inter_area = inter_x*inter_y
	return inter_area / (predict_area+ground_truth_area-inter_area)

def compare(predict_list, ground_truth_list, score_list, match_list):
	'''
	:param predict_list: list, format (x1,y1,x2,y2,c)
	:param ground_truth_list: list, format (x1,y1,x2,y2,c)
	:param score_list:list, store confidence matched
	:param match_list:list, stroe the num of matched boxes
	:return: None
	'''
	ground_truth_unuse = [True for i in range(1, len(ground_truth_list))]
	for j in range(1, len(predict_list)):
		predict_bbox = predict_list[j]
		match = False
		for i in range(1, len(ground_truth_list)):
			if ground_truth_unuse[i-1]:
				# 只寻找一个匹配框,并且是第一个满足条件的,遍历的方式比较慢
				if iou(predict_bbox, ground_truth_list[i])>0.5: # 阈值=0.5
					match = True
					ground_truth_unuse[i-1] = False
					break
		score_list.append(predict_bbox[-1])
		match_list.append(int(match))


if __name__ == '__main__':
	gt_txt = "gt_post.txt"
	pre_txt = "pre_post.txt"
	gt_box_num, gt_dict = get_info(gt_txt)
	pre_box_num, pre_dict = get_info(gt_txt)
	score_list = []
	match_list = []
	print("compare ...")
	for key in pre_dict.keys():
		compare(pre_dict[key],gt_dict[key],score_list,match_list)
	score_match_list = list(zip(score_list, match_list))
	score_match_list.sort(key=lambda x: x[0], reverse=True)
	print('calculate precision/recall...')

	p = list()
	r = list()
	predict_num = 0
	truth_num = 0
	for item in score_match_list:
		predict_num += 1
		truth_num += item[1]
		# recall = TP/(TP + FN)
		r.append(float(truth_num) / gt_box_num)
		# precision = TP/(TP + TN)
		p.append(float(truth_num) / predict_num)

	# precision
	precision = 0
	for pre in p:
		precision += pre
	len_precision = len(p)
	print('precision % =', precision / len_precision)