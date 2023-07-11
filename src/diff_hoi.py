#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import gc

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from argparser import args
from utils import load_res_from_json, load_hico_category, create_gt_2_dets_generator
from plot_hoi import (compute_iou, compute_iou_mat, add_one, check_tpfp,
    add_match_key, plot_single_hoi, plot_multi_hoi, plot_no_hoi)

def _find_matched_pred_id(pred, gt_id):
    """Find matched pair id in prediction using gt id
    """
    # simple version, last one
    # pred_to_gt = {p["match"]: i for i, p in enumerate(pred['hoi_prediction'])}
    # return pred_to_gt[gt_id]
    pred_ids = [p["match"] for p in pred["hoi_prediction"]]
    try:
        return pred_ids.index(gt_id)
    except ValueError:
        return -1

def _get_unmatched_pred(pred):
    """Get unmatched prediction pairs
    """
    return [p for p in pred["hoi_prediction"] if p["match"] is None]

def vis_model_diff(p1, p2, gt, category_info, root_dir, demo_dir):
    '''
    Function: visualize the difference between the prediction of 2 models
    Input:
        -- p1, p2: updated predction with 'match' key
            hoi_prediction --> hoi_annotation [list of dict]
            {
                'subject_id':
                'object_id':
                'category_id':
                'score':
                'match': gt hoi_annotation index or -1
            }
    '''
    file_names_assertion = "[Error] file names should be the same for " \
        "gt, pred1 and pred2, but get {}, {}, and {}".format(
        gt["file_name"], p1["file_name"], p2["file_name"]
    )
    assert p1["file_name"] == p2["file_name"] == gt["file_name"], file_names_assertion
    hoi_c, obj_c, vb_c = category_info
    if not os.path.exists(os.path.join(demo_dir)):
        os.makedirs(os.path.join(demo_dir))

    image_name = p1['file_name']
    print("processing image {}".format(image_name))
    image_path = os.path.join(root_dir, 'images', image_name.split('_')[1], image_name)
    if not os.path.exists(image_path):
        print('image not exist: {}'.format(image_path))
        return

    img = Image.open(image_path)

    # ============== plot pairs in p1 and p2 according to the matching result with gt ==============

    # code logic:
    # for each gt pair, find matched pair in p1 and p2;
    # if not found, plot nothing for that pred; if position matched but classification wrong, plot that pred.
    # save plot into 4 sub dir: yy, yn, ny, nn.

    for gt_id in range(len(gt['hoi_annotation'])):
        gt_pair = gt["hoi_annotation"][gt_id]
        fig = plt.figure(figsize=(10, 10))
        plot_single_hoi(img, gt_pair, gt["annotations"], fig, category_info, subplot_id=224)
        p1_status = "y"
        p2_status = "y"
        p1_id = _find_matched_pred_id(p1, gt_id)
        if p1_id == -1:
            p1_status = "n"
            p1_id = _find_matched_pred_id(p1, - gt_id - 1)
        if p1_id >= 0:
            p1_pair = p1["hoi_prediction"][p1_id]
            plot_single_hoi(img, p1_pair, p1["predictions"], fig, category_info, subplot_id=221)
        else:
            plot_no_hoi(img, fig, subplot_id=221)
        p2_id = _find_matched_pred_id(p2, gt_id)
        if p2_id == -1:
            p2_status = "n"
            p2_id = _find_matched_pred_id(p2, - gt_id - 1)
            if p2_id >= 0:
                print("image name with fp", image_name)
        if p2_id >= 0:
            p2_pair = p2["hoi_prediction"][p2_id]
            plot_single_hoi(img, p2_pair, p2["predictions"], fig, category_info, subplot_id=223)
        else:
            plot_no_hoi(img, fig, subplot_id=223)

        vb_name = vb_c[gt_pair['category_id']-1]
        # sub_name = obj_c[gt['annotations'][gt_pair['subject_id']]['category_id']-1]
        obj_name = obj_c[gt['annotations'][gt_pair['object_id']]['category_id']-1]
        label = '{}-{}'.format(vb_name, obj_name)
        cur_folder = os.path.join(demo_dir, "model_diff", image_name.split(".")[0], p1_status + p2_status)
        if not os.path.exists(cur_folder):
            os.makedirs(cur_folder)
        cur_img_path = os.path.join(cur_folder, "{}_gt_{}_{}.png".format(image_name.split('.')[0], gt_id, label))
        # TODO: result not overridden for now
        # if os.path.exists(cur_img_path):
        #     continue

        height, width = np.asarray(img).shape[:2]
        width *= 2
        height *= 2
        fig.set_size_inches(width/100.0, height/100.0) 
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        # if image too big to save
        try:
            plt.savefig(cur_img_path)
        except ValueError:
            print("image {} not saved successfully due to file size issue.".format(cur_img_path))
        plt.clf()
        plt.close(fig)
        gc.collect()

    # ============== plot unmatched pairs in p1 and p2 ==============
    # code logic:
    # for unpaired pred HOI pairs in p1 and p2,
    # 1. plot them into 2 image separately.
    # or 2. match between p1_unmatched and p2_unmatched and plot result.

    plot_unmatched_separately = True
    fig = plt.figure(figsize=(10, 10))
    p1_unmatched_hois = _get_unmatched_pred(p1)
    p2_unmatched_hois = _get_unmatched_pred(p2)
    # TODO: match p1 with p2
    max_num_pairs = max(len(p1_unmatched_hois), len(p2_unmatched_hois))
    if max_num_pairs > 0:
        height, width = np.array(img).shape[:2]
        if plot_unmatched_separately:
            for i, p1_hoi in enumerate(p1_unmatched_hois):
                plot_single_hoi(img, p1_hoi, p1["predictions"], fig, category_info, subplot_id=None, nrows=max_num_pairs, ncols=2, index=2*i+1)
                # TODO: add text to the plot
            for j, p2_hoi in enumerate(p2_unmatched_hois):
                plot_single_hoi(img, p2_hoi, p2["predictions"], fig, category_info, subplot_id=None, nrows=max_num_pairs, ncols=2, index=2*j+2)
            width *= 2
            height *= max_num_pairs
        else:
            plot_multi_hoi(img, p1_unmatched_hois, p1["predictions"], fig, category_info, subplot_id=121)
            plot_multi_hoi(img, p2_unmatched_hois, p2["predictions"], fig, category_info, subplot_id=122)
            width *= 2
            height *= 1

        # TODO: get_img_name
        cur_folder = os.path.join(demo_dir, "model_diff", image_name.split(".")[0], "unmatched")
        if not os.path.exists(cur_folder):
            os.makedirs(cur_folder)
        cur_img_path = os.path.join(cur_folder, "{}.png".format(image_name.split(".")[0]))

        fig.set_size_inches(width/100.0, height/100.0)
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        try:
            plt.savefig(cur_img_path)
        except ValueError:
            print("image {} not saved successfully due to file size issue.".format(cur_img_path))
        plt.clf()
        plt.close(fig)

    img.close()
    del img

if __name__ == "__main__":
    MODEL_NAME_MAX_LEN = 50
    print("Starting HOI result comparison between 2 models.")
    model1 = args.det_path
    model2 = args.det2_path
    print("Result comparison between {} and {}.".format(model1, model2))
    root_dir = args.root_dir
    assert model2, "[Error]For diff between the results of 2 models, \
        please specify the path of the result of the second model"
    model_name = model1.split("/")[-2][MODEL_NAME_MAX_LEN:] + "-vs-" + model2.split("/")[-2][MODEL_NAME_MAX_LEN:]
    demo_dir = os.path.join(root_dir, model_name)
    gt_result = load_res_from_json(args.gt_path)
    pred_1 = load_res_from_json(model1)
    pred_2 = load_res_from_json(model2)
    updated_pred_1 = add_match_key(gt_result, pred_1)
    updated_pred_2 = add_match_key(gt_result, pred_2)
    category_info = load_hico_category(args.hico_list_dir)
    # s_id, e_id = args.img_ids
    for gt, p1, p2 in create_gt_2_dets_generator(
        gt_result, updated_pred_1, updated_pred_2, args.img_ids
    ):
        vis_model_diff(p1, p2, gt, category_info, root_dir, demo_dir)
    print("HOI results comparison done. See {}".format(demo_dir))
