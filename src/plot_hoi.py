#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# plot HOI detection result

import os
import gc

import numpy as np
# ======= for visualization, uncomment this when you see
# _tkinter.TclError: couldn't connect to display "localhost:10.0"
# import matplotlib
# matplotlib.use("Agg")
# =======
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from utils import (get_iou_np, get_cat_match_mask, get_anno_list, 
    load_hico_category, draw_rec, load_res_from_json, multi_process_image,
    reorganize_func_2nd, create_gt_det_generator,
    mkdir_multiprocess)
from argparser import args

# TODO: some functions similar to those in eval_hico.py, to be removed

def compute_iou(bbox_list1, bbox_list2):
    '''
    Function: get iou mat between gt and det bbox
    '''

    def dict2arr(bbox_list):
        bbox_coord = []
        bbox_cat = []
        for bbox in bbox_list:
            bbox_coord.append(np.array(bbox['bbox']).reshape([1, -1]))
            bbox_cat.append(bbox['category_id'])
        return np.concatenate(bbox_coord, axis=0), np.array(bbox_cat)

    bboxes1, bbox_cat1 = dict2arr(bbox_list1)
    bboxes2, bbox_cat2 = dict2arr(bbox_list2)

    iou_mat = get_iou_np(bboxes1, bboxes2)
    cat_mat = get_cat_match_mask(bbox_cat1, bbox_cat2)
    # cat_mat = np.ones_like(cat_mat)  # TODO: applicable only for VCOCO,
    # # which does not check object category id

    iou_mat = iou_mat * cat_mat
    return iou_mat


def compute_iou_mat(bbox_list1, bbox_list2):
    '''
    Function: match a detected bbox for each gt bbox (iou >= 0.5)
    Input:
        -- bbox_list1/2:
            list of bbox dict
            {
                'bbox':
                'category_id':
            }
    Output:
        -- match_pairs_dict:
            dict with the key of predicted bbox id, the content is a list of the matched gt bbox id
            {
                pred_id: [matched_gt_bbox_id1, matched_gt_bbox_id2, ...]
            }
        -- match_pairs_ov:
            dict with the key of predicted bbox id, the content is a list of the iou with the matched gt bbox
            {
                pred_id: [matched_gt_bbox_iou1, matched_gt_bbox_iou2, ...]
            }
    '''
    iou_mat = compute_iou(bbox_list1, bbox_list2)
    iou_mat_ov = iou_mat.copy()
    iou_mat[iou_mat >= 0.5] = 1
    iou_mat[iou_mat < 0.5] = 0

    match_pairs = np.nonzero(iou_mat)
    match_pairs_dict = {}
    match_pairs_ov = {}
    if iou_mat.max() > 0:
        for i, pred_id in enumerate(match_pairs[1]):
            if pred_id not in match_pairs_dict.keys():
                match_pairs_dict[pred_id] = []
                match_pairs_ov[pred_id] = []
            match_pairs_dict[pred_id].append(match_pairs[0][i])
            match_pairs_ov[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
    return match_pairs_dict, match_pairs_ov


def add_one(prediction):  #Add 1 to all coordinates
    for i, pred_bbox in enumerate(prediction):
        rec = pred_bbox['bbox']
        rec[0]+=1
        rec[1]+=1
        rec[2]+=1
        rec[3]+=1
    return prediction


def check_tpfp(pred_hoi, gt_hoi, match_pairs, pred_bbox, bbox_ov, triplet_list, interaction_list):
    pos_pred_ids = match_pairs.keys()
    vis_tag = np.zeros(len(gt_hoi))
    pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)

    for i, pred_hoi_i in enumerate(pred_hoi):
        pred_hoi_i['match'] = None

    if len(gt_hoi) == 0:
        return
    else:
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                is_match = 0
                is_obj_match = 0
                if isinstance(pred_hoi_i['category_id'], str):
                    pred_hoi_i['category_id'] = int(pred_hoi_i['category_id'].replace('\n', ''))
                # pred bbox are matched (iou) with the gt bbox
                if len(match_pairs) != 0 and pred_hoi_i['subject_id'] in pos_pred_ids and pred_hoi_i['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi_i['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi_i['object_id']]
                    pred_obj_ov = bbox_ov[pred_hoi_i['object_id']]
                    pred_sub_ov = bbox_ov[pred_hoi_i['subject_id']]
                    pred_category_id = pred_hoi_i['category_id']
                    max_ov = 0
                    max_gt_id = 0
                    max_obj_ov = 0
                    max_obj_gt_id = 0
                    # for each predicted hoi, if there are two gt matched, pick the gt with the largest object iou
                    for gt_id in range(len(gt_hoi)):
                        gt_hoi_i = gt_hoi[gt_id]

                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            min_ov_gt = min(pred_sub_ov[pred_sub_ids.index(gt_hoi_i['subject_id'])], pred_obj_ov[pred_obj_ids.index(gt_hoi_i['object_id'])])
                            if min_ov_gt > max_ov:
                                max_ov = min_ov_gt
                                max_gt_id = gt_id

                        elif (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids):
                            is_obj_match = 1
                            min_ov_gt = min(pred_sub_ov[pred_sub_ids.index(gt_hoi_i['subject_id'])], pred_obj_ov[pred_obj_ids.index(gt_hoi_i['object_id'])])
                            if min_ov_gt > max_obj_ov:
                                max_obj_ov = min_ov_gt
                                max_obj_gt_id = gt_id

                if pred_hoi_i['category_id'] not in interaction_list:
                    continue
                # ==== TODO: for V-COCO, comment the following lines ====
                triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'], pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                if triplet not in triplet_list:
                    continue
                # =========
                if is_match == 1:
                    if vis_tag[max_gt_id] == 0:
                        pred_hoi_i['match'] = max_gt_id
                        vis_tag[max_gt_id] = 1
                    else:
                        pred_hoi_i['match'] = - max_gt_id - 1
                        pred_hoi_i['fp_type'] = "dup"
                elif is_obj_match == 1:
                    pred_hoi_i['match'] = - max_obj_gt_id - 1
                    pred_hoi_i['fp_type'] = "cls"
                else:
                    pred_hoi_i['match'] = None
        missed_gt = list(np.where(vis_tag==0)[0])
        return missed_gt

def add_match_key(gt, pred):
    r'''
    Function: check if the predicted hoi is tp or fp; 
                if is tp, set the 'match' key to the gt hoi_annotation index, else set the 'match' key to -1
    Input:
        -- pred: same struct as the gt_annotations
            predictions --> annotations [list of dict]
            {
                'bbox': [x0, y0, x1, y1]
                'category_id': 
                'score':
            }
            hoi_prediction --> hoi_annotation [list of dict]
            {
                'subject_id':
                'object_id':
                'category_id':
                'score':
            }
    Output:
        -- updated_pred:
            hoi_prediction --> hoi_annotation [list of dict]
            {
                'subject_id':
                'object_id':
                'category_id':
                'score':
                'match': gt hoi_annotation index or -1
            }
    '''
    # get triplet and interaction list
    triplet_list, interaction_list = get_anno_list(gt)

    # get the gt file name
    file_name = [gt_i['file_name'] for gt_i in gt]

    for pred_i in pred:
        if pred_i['file_name'] not in file_name or len(pred_i['hoi_prediction'])==0:
            continue
        gt_i = gt[file_name.index(pred_i['file_name'])]
        gt_bbox = gt_i['annotations']
        gt_hoi = gt_i['hoi_annotation']
        pred_hoi = pred_i['hoi_prediction']
        if len(gt_bbox) != 0:
            pred_bbox = add_one(pred_i['predictions'])  # convert zero-based to one-based indices
            bbox_pairs, bbox_ov = compute_iou_mat(gt_bbox, pred_bbox)
            missed_gt = check_tpfp(pred_hoi, gt_hoi, bbox_pairs, pred_bbox, bbox_ov, triplet_list, interaction_list)
            pred_i['missed_gt'] = missed_gt
        else:
            missed_gt = check_tpfp(pred_hoi, gt_hoi, None, None, None, triplet_list, interaction_list)
    return pred

def plot_single_hoi(img, hoi, bboxes, fig, category_info, subplot_id=111, **kwargs):
    r"""Plot an HOI pair `hoi` on a given image `img`, in the region of `fig` specified by `subplot_id`
    """
    _, obj_c, vb_c = category_info
    if subplot_id:
        ax = fig.add_subplot(subplot_id)
    else:
        ax = fig.add_subplot(kwargs["nrows"], kwargs["ncols"], kwargs["index"])
    # sub box in red while obj box in green
    sub_bbox = bboxes[hoi["subject_id"]]["bbox"]
    obj_bbox = bboxes[hoi["object_id"]]["bbox"]
    sub_name = obj_c[bboxes[hoi["subject_id"]]["category_id"]-1]
    obj_name = obj_c[bboxes[hoi["object_id"]]["category_id"]-1]

    subject_score = bboxes[hoi['subject_id']].get('score', 0)
    object_score = bboxes[hoi['object_id']].get('score', 0)
    if subject_score * object_score == 0:
        interaction_score = 0.
    else:
        interaction_score = hoi.get('score', 0) / (subject_score * object_score)
    plt.text(sub_bbox[0], sub_bbox[1], sub_name+' {:.02f}'.format(subject_score * 100.), color='white', fontsize=12)
    plt.text(obj_bbox[0], obj_bbox[1], obj_name+' {:.02f}'.format(object_score * 100.), color='white', fontsize=12)

    ax.add_patch(draw_rec(sub_bbox, 'red'))
    ax.add_patch(draw_rec(obj_bbox, 'green'))

    # draw center line
    sub_center = [sum(sub_bbox[::2])*0.5, sum(sub_bbox[1::2])*0.5]
    obj_center = [sum(obj_bbox[::2])*0.5, sum(obj_bbox[1::2])*0.5]

    plt.plot([sub_center[0], obj_center[0]], [sub_center[1], obj_center[1]], color="tab:blue")
    hoi_text = vb_c[hoi['category_id']-1] + '{:.02f}'.format(interaction_score * 100.)
    binary_score = hoi.get("binary_score")
    if binary_score is not None:
        hoi_text += " {:.02f}".format(binary_score * 100)
    plt.text((sub_center[0]+obj_center[0])*0.5, (sub_center[1]+obj_center[1])*0.5, hoi_text, color='white', fontsize=12)

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

def plot_no_hoi(img, fig, subplot_id=111, **kwargs):
    r"""Plot in a specified subplot with no additional annotations
    """
    if subplot_id:
        ax = fig.add_subplot(subplot_id)
    else:
        ax = fig.add_subplot(kwargs["nrows"], kwargs["ncols"], kwargs["index"])

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

def plot_multi_hoi(img, hois, bboxes, fig, category_info, subplot_id=111, **kwargs):
    r"""Plot multiple HOI pairs `hois` on a given image `img`,
    in the region of `fig` specified by `subplot_id`
    """
    _, obj_c, vb_c = category_info
    if subplot_id:
        ax = fig.add_subplot(subplot_id)
    else:
        ax = fig.add_subplot(kwargs["nrows"], kwargs["ncols"], kwargs["index"])
    # sub box in red while obj box in green
    for hoi in hois:
        sub_bbox = bboxes[hoi["subject_id"]]["bbox"]
        obj_bbox = bboxes[hoi["object_id"]]["bbox"]
        sub_name = obj_c[bboxes[hoi["subject_id"]]["category_id"]-1]
        obj_name = obj_c[bboxes[hoi["object_id"]]["category_id"]-1]

        subject_score = bboxes[hoi['subject_id']].get('score', 0)
        object_score = bboxes[hoi['object_id']].get('score', 0)
        if subject_score * object_score == 0:
            interaction_score = 0.
        else:
            interaction_score = hoi.get('score', 0) / (subject_score * object_score)
        plt.text(sub_bbox[0], sub_bbox[1], sub_name+' {:.02f}'.format(subject_score * 100.), color='white', fontsize=12)
        plt.text(obj_bbox[0], obj_bbox[1], obj_name+' {:.02f}'.format(object_score * 100.), color='white', fontsize=12)

        ax.add_patch(draw_rec(sub_bbox, 'red'))
        ax.add_patch(draw_rec(obj_bbox, 'green'))

        # draw center line
        sub_center = [sum(sub_bbox[::2])*0.5, sum(sub_bbox[1::2])*0.5]
        obj_center = [sum(obj_bbox[::2])*0.5, sum(obj_bbox[1::2])*0.5]

        plt.plot([sub_center[0], obj_center[0]], [sub_center[1], obj_center[1]], color="tab:blue")
        plt.text((sub_center[0]+obj_center[0])*0.5, (sub_center[1]+obj_center[1])*0.5, vb_c[hoi['category_id']-1] + '{:.02f}'.format(interaction_score * 100.), color='white', fontsize=12)

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)


def vis_hoi_tpfp_by_img(pred, gt, category_info, root_dir, demo_dir, folder_name="hoi_analysis"):
    r'''Visualize the HOI pairs in each image one by one,
    and distingush the types (tp/fp/false/missed) at the same time.
    Args:
        -- pred: updated predction with 'match' key
            hoi_prediction --> hoi_annotation [list of dict]
            {
                'subject_id':
                'object_id':
                'category_id':
                'score':
                'match': gt hoi_annotation index or -1
            }
    '''
    assert pred["file_name"] == gt["file_name"], "pred {} and gt {} not matched".format(pred["file_name"], gt["file_name"])
    hoi_c, obj_c, vb_c = category_info

    image_name = pred['file_name']
    print("processing image {}".format(image_name))
    image_path = os.path.join(root_dir, 'images', image_name.split('_')[1], image_name)
    if not os.path.exists(image_path):
        print('image not exist: {}'.format(image_path))
        return

    img = Image.open(image_path)

    for i in range(len(pred['hoi_prediction'])):
        hoi = pred['hoi_prediction'][i]
        if hoi['score'] < 0.1:
            continue

        vb_name = vb_c[hoi['category_id']-1]
        # sub_name = obj_c[pred['predictions'][hoi['subject_id']]['category_id']-1]
        obj_name = obj_c[pred['predictions'][hoi['object_id']]['category_id']-1]
        label = '{}-{}'.format(vb_name, obj_name)
        subject_score = pred['predictions'][hoi['subject_id']]['score']
        object_score = pred['predictions'][hoi['object_id']]['score']
        interaction_score = hoi['score'] / (subject_score * object_score)
        score = '{:.02f}-{:.02f}-{:.02f}'.format(subject_score * 100., object_score * 100., interaction_score * 100.)
        cur_path = os.path.join(demo_dir, folder_name, image_name.split(".")[0])
        if hoi['match'] is None:
            cur_path = os.path.join(cur_path, 'false')
        elif hoi['match'] < 0 and hoi['fp_type'] == 'dup':
            cur_path = os.path.join(cur_path, 'fp_dup')
        elif hoi['match'] < 0 and hoi['fp_type'] == 'cls':
            cur_path = os.path.join(cur_path, 'fp_cls')
        else:
            cur_path = os.path.join(cur_path, 'tp')
        mkdir_multiprocess(cur_path)
        cur_img_path = os.path.join(cur_path, '{}_{}_{}_{}.png'.format(image_name.split('.')[0], i, label, score))
        if os.path.exists(cur_img_path):
            continue

        fig = plt.figure(figsize=(10, 10))
        if hoi['match'] is None:
            plot_single_hoi(img, hoi, pred['predictions'], fig, category_info, subplot_id=111)
            height, width = np.asarray(img).shape[:2]
        elif hoi['match'] < 0:
            plot_single_hoi(img, hoi, pred['predictions'], fig, category_info, subplot_id=121)
            gt_id = -hoi['match']-1
            plot_single_hoi(img, gt['hoi_annotation'][gt_id], gt['annotations'], fig, category_info, subplot_id=122)
            height, width = np.asarray(img).shape[:2]
            width *= 2
        else:
            plot_single_hoi(img, hoi, pred['predictions'], fig, category_info, subplot_id=121)
            gt_id = hoi['match']
            plot_single_hoi(img, gt['hoi_annotation'][gt_id], gt['annotations'], fig, category_info, subplot_id=122)
            height, width = np.asarray(img).shape[:2]
            width *= 2
        fig.set_size_inches(width/100.0, height/100.0) 
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.savefig(cur_img_path)
        plt.clf()
        plt.close(fig)
        gc.collect()

    # Draw missed gt
    if 'missed_gt' in pred and len(pred['missed_gt']) > 0:
        miss_folder = os.path.join(demo_dir, 'hoi_analysis', image_name.split('.')[0], 'missed_gt')
        mkdir_multiprocess(miss_folder)
        for i in pred['missed_gt']:
            hoi = gt['hoi_annotation'][i]
            fig = plt.figure(figsize=(10, 10))
            vb_name = vb_c[hoi['category_id']-1]
            # sub_name = obj_c[gt['annotations'][hoi['subject_id']]['category_id']-1]
            obj_name = obj_c[gt['annotations'][hoi['object_id']]['category_id']-1]
            # miss_folder = os.path.join(demo_dir, 'interactions', '{}_{}'.format(vb_name, obj_name), 'missed_gt')
            # mkdir_multiprocess(miss_folder)
            label = '{}-{}'.format(vb_name, obj_name)
            cur_img_path = os.path.join(miss_folder, '{}_{}_{}.png'.format(image_name.split('.')[0], i, label))
            plot_single_hoi(img, hoi, gt['annotations'], fig, category_info, subplot_id=111)
            height, width = np.asarray(img).shape[:2]
            fig.set_size_inches(width/100.0, height/100.0) 
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
            plt.savefig(cur_img_path)
            plt.clf()
            plt.close(fig)
            gc.collect()
    img.close()
    del img

def vis_hoi_tpfp_by_category(pred, gt, vb_id, obj_id, category_info, root_dir, demo_dir, lower=-1000, upper=1000):
    r'''Group the predictions by verb and obj categories,
    and distingush the tp/fp/false at the same time.
    Args:
        -- pred: updated predction with 'match' key
            hoi_prediction --> hoi_annotation [list of dict]
            {
                'subject_id':
                'object_id':
                'category_id':
                'score':
                'match': gt hoi_annotation index or -1
            }
    '''
    # code logic:
    # go through det_result, find det and gt for each category
    # for each det, for each hoi pair, plot 1) the pair, 2) all det, and 3) all gt
    assert pred["file_name"] == gt["file_name"], "pred {} and gt {} not matched".format(pred["file_name"], gt["file_name"])
    _, obj_c, vb_c = category_info

    # folder_name = "hoi_analysis_classwise"
    image_name = pred['file_name']
    # print("processing image {}".format(image_name))
    image_path = os.path.join(root_dir, 'images', image_name.split('_')[1], image_name)
    if not os.path.exists(image_path):
        print('image not exist: {}'.format(image_path))
        return

    img = Image.open(image_path)

    for i in range(len(pred['hoi_prediction'])):
        hoi = pred['hoi_prediction'][i]
        # only plot when vb_id and obj_id are as required
        obj_box = pred['predictions'][hoi['object_id']]
        if hoi["category_id"] != vb_id or (obj_id is not None and obj_box["category_id"] != obj_id):
            continue

        vb_name = vb_c[vb_id - 1]
        # sub_name = obj_c[pred['predictions'][hoi['subject_id']]['category_id'] - 1]
        obj_name = obj_c[obj_box["category_id"] - 1]
        label = '{}-{}'.format(vb_name, obj_name)
        subject_score = pred['predictions'][hoi['subject_id']]['score']
        object_score = pred['predictions'][hoi['object_id']]['score']
        interaction_score = hoi['score'] / (subject_score * object_score)
        score = '{:.02f}-{:.02f}-{:.02f}'.format(subject_score * 100., object_score * 100., interaction_score * 100.)
        # for False and FP HOI, only plot when its score is in certain range
        if (hoi["match"] is None or hoi["match"] < 0) and (interaction_score < lower or interaction_score > upper):
            continue
        # cur_path = os.path.join(demo_dir, folder_name, vb_name, obj_name)
        cur_path = demo_dir
        if hoi['match'] is None:
            cur_path = os.path.join(cur_path, 'false')
        elif hoi['match'] < 0:
            cur_path = os.path.join(cur_path, 'fp')
        else:
            cur_path = os.path.join(cur_path, 'tp')
        mkdir_multiprocess(cur_path)
        cur_img_path = os.path.join(cur_path, '{}_{}_{}_{}.png'.format(image_name.split('.')[0], i, label, score))
        if os.path.exists(cur_img_path):
            continue

        fig = plt.figure(figsize=(10, 10))
        if hoi['match'] is None:
            # false
            plot_single_hoi(img, hoi, pred['predictions'], fig, category_info, subplot_id=231)
        elif hoi['match'] < 0:
            # fp
            plot_single_hoi(img, hoi, pred['predictions'], fig, category_info, subplot_id=231)
        else:
            # tp
            plot_single_hoi(img, hoi, pred['predictions'], fig, category_info, subplot_id=231)
        # plot all det
        plot_multi_hoi(img, pred["hoi_prediction"], pred["predictions"], fig, category_info, subplot_id=232)
        show_anno_in_plot(img, pred["hoi_prediction"], pred["predictions"], fig, category_info, subplot_id=235)
        # plot all gt
        plot_multi_hoi(img, gt["hoi_annotation"], gt["annotations"], fig, category_info, subplot_id=233)
        show_anno_in_plot(img, gt["hoi_annotation"], gt["annotations"], fig, category_info, subplot_id=236)
        height, width = np.asarray(img).shape[:2]
        width *= 3
        height *= 2
        fig.set_size_inches(width/100.0, height/100.0)
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.savefig(cur_img_path)
        plt.clf()
        plt.close(fig)
        gc.collect()

    img.close()
    del img

def show_anno_in_plot(img, hois, bboxes, fig, category_info, subplot_id, **kwargs):
    r"""Show the annotations/predicitons of multiple HOI pairs `hois`,
    in the region of `fig` specified by `subplot_id`
    """
    _, obj_c, vb_c = category_info
    if subplot_id:
        ax = fig.add_subplot(subplot_id)
    else:
        ax = fig.add_subplot(kwargs["nrows"], kwargs["ncols"], kwargs["index"])
    loc_x, loc_y = 10., 20.
    for hoi in hois:
        sub_name = obj_c[bboxes[hoi["subject_id"]]["category_id"]-1]
        obj_name = obj_c[bboxes[hoi["object_id"]]["category_id"]-1]
        vb_name = vb_c[hoi['category_id']-1]

        subject_score = bboxes[hoi['subject_id']].get('score', 0)
        object_score = bboxes[hoi['object_id']].get('score', 0)
        if subject_score * object_score == 0:
            interaction_score = 0.
        else:
            interaction_score = hoi.get('score', 0) / (subject_score * object_score)
        plt.text(loc_x, loc_y, '{}-{}-{}:{:.02f}-{:.02f}-{:.02f}'.format(sub_name, vb_name, obj_name, subject_score * 100., interaction_score * 100., object_score * 100.), color='white', fontsize=12)
        loc_y += 20.

    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.zeros_like(img))

if __name__ == "__main__":
    multi_process = True
    print("Starting HOI result plotting.")
    root_dir = args.root_dir
    model_name = args.det_path.split("/")[-2]
    demo_dir = os.path.join(root_dir, model_name)
    gt_result = load_res_from_json(args.gt_path)
    pred_result = load_res_from_json(args.det_path)
    updated_pred = add_match_key(gt_result, pred_result)
    category_info = load_hico_category(args.hico_list_dir)
    hoi_c, obj_c, vb_c = category_info
    # feature `vis_hoi`:
    # visuaize HOI (FP, TP, False, Missed) by image ids.
    # s_id, e_id = args.img_ids
    gt_det_generator = create_gt_det_generator(gt_result, updated_pred, args.img_ids)
    for gt, det in tqdm(list(gt_det_generator)):
        if len(det['hoi_prediction']) == 0:
            continue
        vis_hoi_tpfp_by_img(det, gt, category_info, root_dir, demo_dir)
    print("HOI result visualization done. See {}".format(demo_dir))
