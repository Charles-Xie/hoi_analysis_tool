#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   detection.py
@Time    :   2020/09/01 14:24:02
@Author  :   HuYue
@Contact :   huyue@megvii.com
@Desc    :   None
'''

# plot object detection result

import os
import math

import numpy as np
# ======= for visualization, uncomment this when you see
# _tkinter.TclError: couldn't connect to display "localhost:10.0"
# import matplotlib
# matplotlib.use("Agg")
# =======
from matplotlib import pyplot as plt
from PIL import Image

from utils import (load_hico_category, draw_rec, 
    multi_process_image, get_name, load_res_from_json,
    reorganize_func_1st, create_gt_det_generator)
from argparser import args


def vis_hoi_in_whole_img(data, category_info, root_dir, demo_dir, mode='gt'):
    r"""[Warning] Deprecated. Not used currently.
        Visualize all HOI pairs in one image.
        Originally `vis_whole_img` function.
    """
    _, obj_c, vb_c = category_info
    annotations = 'annotations' if mode == 'gt' else 'predictions'
    hoi_annotation = 'hoi_annotation' if mode == 'gt' else 'hoi_prediction'
    folder_name = 'gt_whole' if mode == 'gt' else 'det_whole'

    image_name = data['file_name']
    image_path = os.path.join(root_dir, 'images', image_name.split('_')[1], image_name)
    if not os.path.exists(image_path):
        print('image not exist: {}'.format(image_path))
        return

    img = Image.open(image_path)

    cur_folder = os.path.join(demo_dir, folder_name, image_name.split('_')[1])
    if not os.path.exists(cur_folder):
        os.makedirs(cur_folder)

    cur_img_path = os.path.join(cur_folder, '{}.png'.format(get_name(image_name)))
    if os.path.exists(cur_img_path):
        return
    
    fig = plt.figure(figsize=(10, 10))

    # hoi_ids = [hoi['category_id'] for hoi in data[hoi_annotation]]
    # from collections import Counter
    # c = Counter(hoi_ids)
    # most_freq_id = c.most_common(1)[0][0]
    for i in range(len(data[hoi_annotation])):
        ax = plt.gca()
        hoi = data[hoi_annotation][i]

        # if hoi['category_id'] != most_freq_id:
        #     continue
        # if hoi.get('score', 1) < 0.3:
        #     continue
        # # if hoi['category_id'] != 31:
        # #     continue
        vb_name = vb_c[hoi['category_id']-1]
        sub_name = obj_c[data[annotations][hoi['subject_id']]['category_id']-1]
        obj_name = obj_c[data[annotations][hoi['object_id']]['category_id']-1]
        label = '{}-{}'.format(vb_name, obj_name)
        print(image_name, i, label)

        # sub box in red while obj box in green
        sub_bbox = data[annotations][hoi['subject_id']]['bbox']
        obj_bbox = data[annotations][hoi['object_id']]['bbox']
        obj_bbox_nan = math.isnan(obj_bbox[0])  # FIXME: handle nan box?

        # plt.text(sub_bbox[0], sub_bbox[1], sub_name, color='white', fontsize=12)
        ax.add_patch(draw_rec(sub_bbox, 'red', linewidth=2))
        # if not obj_bbox_nan:  # handle object boxes with NaN coordinates
        # plt.text(obj_bbox[0], obj_bbox[1], obj_name, color='white', fontsize=12)
        ax.add_patch(draw_rec(obj_bbox, 'green', linewidth=2))

        # draw center line
        sub_center = [sum(sub_bbox[::2])*0.5, sum(sub_bbox[1::2])*0.5]
        obj_center = [sum(obj_bbox[::2])*0.5, sum(obj_bbox[1::2])*0.5]

        plt.plot([sub_center[0], obj_center[0]], [sub_center[1], obj_center[1]], linewidth=2)
        # plt.text((sub_center[0]+obj_center[0])*0.5, (sub_center[1]+obj_center[1])*0.5, vb_c[hoi['category_id']-1] + '{:.02f}'.format(hoi.get('score', 0)), color='white', fontsize=12)
    height = np.asarray(img).shape[0]
    width = np.asarray(img).shape[1]
    fig.set_size_inches(width/100.0, height/100.0) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.imshow(img)
    # plt.title("{}-{}".format(image_name, folder_name), y=0)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(cur_img_path)
    plt.close()
    img.close()

def vis_box_in_img(data, category_info, root_dir, demo_dir, mode='gt', det_vis_thres=None):
    r"""Visuaize bounding boxes in a image.
    """
    _, obj_c, _ = category_info
    annotations = 'annotations' if mode == 'gt' else 'predictions'
    folder_name = 'gt_box' if mode == 'gt' else  'det_box'

    image_name = data['file_name']
    image_path = os.path.join(root_dir, 'images', image_name.split('_')[1], image_name)
    if not os.path.exists(image_path):
        print('image not exist: {}'.format(image_path))
        return

    img = Image.open(image_path)

    cur_folder = os.path.join(demo_dir, folder_name, image_name.split('_')[1])
    if not os.path.exists(cur_folder):
        os.makedirs(cur_folder)

    cur_img_path = os.path.join(cur_folder, '{}.png'.format(get_name(image_name)))
    if os.path.exists(cur_img_path):
        return

    fig = plt.figure(figsize=(10, 10))

    for box in data[annotations]:
        box_score = box.get("score", 0)
        if det_vis_thres and 0 < box_score <= det_vis_thres:
            continue
        ax = plt.gca()
        box_name = obj_c[box['category_id']-1]
        print(image_name, box_name)

        # sub box in red while obj box in green
        bbox = box['bbox']
        if math.isnan(bbox[0]):
            continue
        plt.text(bbox[0], bbox[1], box_name+' {:.02f}'.format(box_score * 100), color='white', fontsize=10)
        if box_name == "person":
            ax.add_patch(draw_rec(bbox, "red"))
        else:
            ax.add_patch(draw_rec(bbox, "green"))
    height = np.asarray(img).shape[0]
    width = np.asarray(img).shape[1]
    fig.set_size_inches(width/100.0, height/100.0) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.imshow(img)
    plt.title("{}-{}".format(image_name, folder_name), y=0)
    plt.xticks([])
    plt.yticks([])
        
    plt.savefig(cur_img_path)
    plt.close()
    img.close()

# TODO: function duplicates with plot_multi_hoi in plot_hoi.py
def vis_hoi_in_img(data, category_info, root_dir, demo_dir, mode='gt'):
    r"""Visualize hoi pairs, one pair in each image.
    """
    _, obj_c, vb_c = category_info
    annotations = 'annotations' if mode == 'gt' else 'predictions'
    hoi_annotation = 'hoi_annotation' if mode == 'gt' else 'hoi_prediction'
    folder_name = 'gt_hoi' if mode == 'gt' else  'det_hoi'

    image_name = data['file_name']
    image_path = os.path.join(root_dir, 'images', image_name.split('_')[1], image_name)
    if not os.path.exists(image_path):
        print('image not exist: {}'.format(image_path))
        return

    img = Image.open(image_path)

    for i in range(len(data[hoi_annotation])):
        cur_folder = os.path.join(demo_dir, folder_name, image_name.split('_')[1], image_name.split('.')[0])
        if not os.path.exists(cur_folder):
            os.makedirs(cur_folder)

        hoi = data[hoi_annotation][i]
        if hoi.get('score', 1) < 0.05:
            continue

        vb_name = vb_c[hoi['category_id']-1]
        sub_det = data[annotations][hoi['subject_id']]
        obj_det = data[annotations][hoi['object_id']]
        sub_name = obj_c[sub_det['category_id']-1]
        obj_name = obj_c[obj_det['category_id']-1]
        label = '{}-{}'.format(vb_name, obj_name)
        print(image_name, i, label)

        cur_img_path = os.path.join(cur_folder, '{}_{}_{}.png'.format(get_name(image_name), i, label))
        if hoi.get('score', 1) < 1:
            cur_img_path = os.path.join(cur_folder, '{}_{}_{}_{}.png'.format(get_name(image_name), i, label, hoi.get('score', 1)))

        if os.path.exists(cur_img_path):
            continue

        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # sub box in red while obj box in green
        sub_bbox = sub_det['bbox']
        obj_bbox = obj_det['bbox']

        if math.isnan(obj_bbox[0]):
            obj_bbox = [0, 0 ,1, 1]
            # if obj box is NaN, set it to be the upper-left corner point

        # plt.text(sub_bbox[0], sub_bbox[1], "{}".format(sub_name), color='white', fontsize=12)
        # plt.text(obj_bbox[0], obj_bbox[1], "{}".format(obj_name), color='white', fontsize=12)
        ax.add_patch(draw_rec(sub_bbox, 'red', linewidth=4))
        ax.add_patch(draw_rec(obj_bbox, 'green', linewidth=4))

        # draw center line
        sub_center = [sum(sub_bbox[::2])*0.5, sum(sub_bbox[1::2])*0.5]
        obj_center = [sum(obj_bbox[::2])*0.5, sum(obj_bbox[1::2])*0.5]

        plt.plot([sub_center[0], obj_center[0]], [sub_center[1], obj_center[1]], linewidth=4)
        if hoi["category_id"] >= 0:
            vb_name = vb_c[hoi['category_id']-1]
        else:
            vb_name = "unknown"
        hoi_text = vb_name
        binary_score = hoi.get("binary_score")
        if binary_score is not None:
            hoi_text += " {:.02f}".format(binary_score * 100)
        # plt.text((sub_center[0]+obj_center[0])*0.5, (sub_center[1]+obj_center[1])*0.5, hoi_text, color='white', fontsize=12)

        height, width = np.asarray(img).shape[:2]
        fig.set_size_inches(width/100.0, height/100.0) 
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
        plt.imshow(img)
        # plt.title("{}-{}-{}-{}".format(image_name, folder_name, i, label), y=0)
        plt.xticks([])
        plt.yticks([])

        plt.savefig(cur_img_path)
        plt.close()
    img.close()

def concat_img(image_name, demo_dir):
    r"""Concat det image with gt image.
    """
    folders = ['det_box', 'gt_box' ]

    cur_folder = os.path.join(demo_dir, 'gt_det_box_diff', image_name.split('_')[1])
    if not os.path.exists(cur_folder):
        os.makedirs(cur_folder)
    cur_img_path = os.path.join(cur_folder, '{}.png'.format(get_name(image_name)))
    if os.path.exists(cur_img_path):
        return

    imgs = []
    for i in range(len(folders)):
        folder_name = folders[i]
        image_path = os.path.join(os.path.join(demo_dir, folder_name, image_name.split('_')[1]), '{}.png'.format(get_name(image_name)))
        img = Image.open(image_path)
        imgs.append(np.asarray(img))
        img.close()
    imgs = np.concatenate(imgs, axis=1)

    fig = plt.figure(figsize=(15, 10))
    height, width = imgs.shape[:2]
    fig.set_size_inches(width/100.0, height/100.0) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.imshow(imgs)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(cur_img_path)
    plt.close()

if __name__ == '__main__':
    print("Starting detection result plotting.")
    # data loading
    gt_result = load_res_from_json(args.gt_path)
    det_result = load_res_from_json(args.det_path)
    det_names = args.det_path.split("/")
    model_name = "{}_{}".format(det_names[-2], det_names[-1].split(".")[-2])
    root_dir = args.root_dir
    demo_dir = os.path.join(root_dir, model_name)
    category_info = load_hico_category(args.hico_list_dir)

    # detection result ploting
    # s_id, e_id = args.img_ids
    det_file_names = [det["file_name"] for det in det_result]
    gt_det_generator = create_gt_det_generator(gt_result, det_result, args.img_ids)
    for gt, det in gt_det_generator:
        # vis_box_in_img(gt, category_info, root_dir, demo_dir, mode='gt', det_vis_thres=args.det_vis_thres)
        # vis_box_in_img(det, category_info, root_dir, demo_dir, mode='det', det_vis_thres=args.det_vis_thres)
        # concat_img(gt["file_name"], demo_dir)
        # visualize HOI pairs
        if "hoi_annotation" in gt:
            vis_hoi_in_img(gt, category_info, root_dir, demo_dir, mode="gt")
            # vis_hoi_in_whole_img(gt, category_info, root_dir, demo_dir, mode="gt")
        else:
            print("[Warning]hoi_annotation not found for image {}".format(gt["file_name"]))
        if "hoi_prediction" in det:
            vis_hoi_in_img(det, category_info, root_dir, demo_dir, mode="det")
            # pass
        else:
            print("[Warning]hoi_prediction not found for image {}".format(det["file_name"]))

        # multi-process processing to speed up
        # multi_process_image(vis_hoi_in_img, gt, reorganize_func_1st, 16,
        #     category_info, root_dir, demo_dir, "gt")
        # multi_process_image(vis_hoi_in_img, det, reorganize_func_1st, 16,
        #     category_info, root_dir, demo_dir, "det")
    print("Detection result visualization done. See {}".format(demo_dir))
    # TODO: add simple gt/pred plotting support
