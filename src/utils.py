#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/09/01 17:34:23
@Author  :   HuYue
@Contact :   huyue@megvii.com
@Desc    :   None
'''

import os
import time
import json
from multiprocessing import Process
from collections import OrderedDict
import errno

import numpy as np
import matplotlib.patches as patches
from tqdm import tqdm

#####################################################################
# --------------------------- Dataset Stats --------------------------- #
#####################################################################

VALID_OBJ_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90]

VALID_ACTION_IDS_VCOCO = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 28]

#####################################################################
# --------------------------- Box Utils --------------------------- #
#####################################################################

def get_iou_np(boxes1, boxes2):
    r"""Get of two box sets.
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = np.split(boxes1[:, :4], 4, axis=1)
    b2_x0, b2_y0, b2_x1, b2_y1 = np.split(boxes2[:, :4], 4, axis=1)

    area1 = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    area2 = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    x0 = np.maximum(b1_x0, np.transpose(b2_x0))
    y0 = np.maximum(b1_y0, np.transpose(b2_y0))

    w0 = np.minimum(b1_x1, np.transpose(b2_x1)) - x0
    h0 = np.minimum(b1_y1, np.transpose(b2_y1)) - y0
    w0 = np.maximum(w0, 0)
    h0 = np.maximum(h0, 0)
    inter = w0 * h0
    return inter / (area1 + np.transpose(area2) - inter)


def get_cat_match_mask(cat1, cat2):
    r'''Get match mask for two category array
    test case:
        cat1 = np.array(range(5))
        cat2 = np.array([1, 1, 3, 4])
        print(get_cat_match_mask(cat1, cat2))
    '''
    N1 = cat1.shape[0]
    N2 = cat2.shape[0]
    cat1 = np.repeat(np.reshape(cat1, [N1, 1]), N2, axis=1)
    cat2 = np.repeat(np.reshape(cat2, [1, N2]), N1, axis=0)
    mask = (cat1 == cat2) * 1.0
    return mask


def get_ioa_np(boxes1, boxes2):
    b1_x0, b1_y0, b1_x1, b1_y1 = np.split(boxes1[:, :4], 4, axis=1)
    b2_x0, b2_y0, b2_x1, b2_y1 = np.split(boxes2[:, :4], 4, axis=1)

    area1 = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    x0 = np.maximum(b1_x0, np.transpose(b2_x0))
    y0 = np.maximum(b1_y0, np.transpose(b2_y0))

    w0 = np.minimum(b1_x1, np.transpose(b2_x1)) - x0
    h0 = np.minimum(b1_y1, np.transpose(b2_y1)) - y0
    w0 = np.maximum(w0, 0)
    h0 = np.maximum(h0, 0)
    inter = w0 * h0
    return inter / np.maximum(1.0, area1)


#####################################################################
# -------------------------- Data Utils --------------------------- #
#####################################################################

def load_json(_path):
    with open(_path, 'r') as f:
        data = json.load(f)
    return data


def txt2list(_path):
    with open(_path, 'r') as f:
        cur_data = f.readlines()
    category = []
    for i in range(2, len(cur_data)):
        tmp = cur_data[i].strip().split()
        if len(tmp) > 2:
            category.append(' '.join(tmp[::-1][:2]))
        elif len(tmp) == 2:
            category.append(tmp[1])
        else:
            category.append(tmp[0])
    return category

def sim_txt2list(_path):
    with open(_path, 'r') as f:
        cur_data = f.readlines()
    category = []
    for i in range(2, len(cur_data)):
        category.append(cur_data[i].strip())
    return category


def load_hico_category(hico_list_dir):
    r'''Map category_id to category_name for HICO_DET dataset

    Args:
        hico_list_dir (str): path for dir where files like
        `hico_list_hoi.txt` exist
    Return:
        hoi_c: hoi name list in the order of category_id
        obj_c: object name list in the order of category_id
        vb_c: action name list in the order of category_id
    '''

    # load category
    hoi_category = os.path.join(hico_list_dir, 'hico_list_hoi.txt')
    # obj_category = os.path.join(hico_list_dir, 'hico_list_obj.txt')
    obj_category = os.path.join(hico_list_dir, 'hico_list_91obj.txt')
    vb_category = os.path.join(hico_list_dir, 'hico_list_vb.txt')
    hoi_c = txt2list(hoi_category)
    obj_c = sim_txt2list(obj_category)
    vb_c = txt2list(vb_category)
    if len(vb_c) == 29:
        # v-coco dataset vb list from 0 to 28
        # move elements one step left
        # move first element (idx 0) to last (idx -1)
        vb_c = vb_c[1:] + vb_c[:1]
    return hoi_c, obj_c, vb_c

def list2dic(category_list):
    name2id = OrderedDict()
    id2name = OrderedDict()
    for i in range(len(category_list)):
        name2id[category_list[i]] = i + 1
        id2name[i+1] = category_list[i]
    return name2id, id2name


def get_anno_list(gt):
    triplet_list = []
    interaction_list = []
    for gt_i in gt:
        gt_hoi = gt_i['hoi_annotation']
        gt_bbox = gt_i['annotations']
        for gt_hoi_i in gt_hoi:
            triplet = [gt_bbox[gt_hoi_i['subject_id']]['category_id'], gt_bbox[gt_hoi_i['object_id']]['category_id'], gt_hoi_i['category_id']]
            interaction = gt_hoi_i['category_id']
            if triplet not in triplet_list:
                triplet_list.append(triplet)
            if interaction not in interaction_list:
                interaction_list.append(interaction)
    return triplet_list, interaction_list

def transform_category_ids_90(result):
    r"""if values of category_id in detection/gt result are in range [1, 80],
    transform to [1, 90].
    .. note::
        this is in-place transformation for `result`
    Args:
        result (list of dict): gt/det result read from json
    Return:
        a list of dict, which is `result` after transformation
    """
    valid_ids = VALID_OBJ_IDS

    if "predictions" in result[0]:
        key = "predictions"
    else:
        key = "annotations"
    vb_key = None
    if "hoi_prediction" in result[0]:
        vb_key = "hoi_prediction"
    if "hoi_annotation" in result[0]:
        vb_key = "hoi_annotation"
    if vb_key:
        pred_vb_category_ids = set([hoi["category_id"] for img_det in result for hoi in img_det[vb_key]])
        print("{} has {} verb categories from {} to {}".format(
            vb_key, len(pred_vb_category_ids), min(pred_vb_category_ids), max(pred_vb_category_ids)
        ))
        assert len(pred_vb_category_ids) <= 117, "{} has more than 117 vb categories ({})".format(
                vb_key, len(pred_vb_category_ids)
            )
    pred_obj_category_ids = set([pred["category_id"] for img_det in result for pred in img_det[key]])
    print("{} has {} object categories from {} to {}".format(
        key, len(pred_obj_category_ids), min(pred_obj_category_ids), max(pred_obj_category_ids)
    ))
    assert len(pred_obj_category_ids) <= 80, "{} has more than 80 obj categories ({})".format(
            key, len(pred_obj_category_ids)
        )
    if max(pred_obj_category_ids) == 80:
        # need to map obj categories 1->80 to 1->90
        print("Mapping obj categories 1->80 to 1->90")
        for img_det in result:
            for pred in img_det[key]:
                pred["category_id"] = valid_ids[pred["category_id"] - 1]
    return result

def load_res_from_json(json_path):
    r"""Load gt/det result from our standard format json.

    Args:
        json_path: path of json file
    Return:
        a list of dict which is the gt/det result after processing
    """
    print("Loading gt/det file from {}...".format(json_path))
    res = load_json(json_path)
    # return transform_category_ids_90(res)
    return res


def create_gt_det_generator(gt_result, det_result, img_ids):
    r"""create a generator which yields paired gt and det
    """
    gt_file_names = [gt["file_name"] for gt in gt_result]
    det_file_names = [det["file_name"] for det in det_result]
    if len(img_ids) != 2:
        for img_id in img_ids:
            file_name = 'HICO_test2015_{:08d}.jpg'.format(img_id)
            # file_name = 'COCO_val2014_{:012d}.jpg'.format(img_id)
            det = det_result[det_file_names.index(file_name)]
            gt = gt_result[gt_file_names.index(file_name)]
            yield gt, det
    else:
        start_id, end_id = img_ids
        for i in range(start_id, end_id):
            gt = gt_result[i]
            # if det_result[i]["file_name"] == gt["file_name"]:
            #     det = det_result[i]
            # else:
            try:
                det_idx = det_file_names.index(gt["file_name"])
                det = det_result[det_idx]
            except ValueError:
                continue
            yield gt, det

def create_gt_2_dets_generator(gt_result, det1_result, det2_result, img_ids):
    r"""create a generator which yields paired gt and det1, det2 for 2 models
    """
    gt_file_names = [gt["file_name"] for gt in gt_result]
    det1_file_names = [det1["file_name"] for det1 in det1_result]
    det2_file_names = [det2["file_name"] for det2 in det2_result]
    if len(img_ids) != 2:
        for img_id in img_ids:
            file_name = 'HICO_test2015_{:08d}.jpg'.format(img_id)
            det1 = det1_result[det1_file_names.index(file_name)]
            det2 = det2_result[det2_file_names.index(file_name)]
            gt = gt_result[gt_file_names.index(file_name)]
            yield gt, det1, det2
    else:
        start_id, end_id = img_ids
        for i in range(start_id, end_id):
            gt = gt_result[i]
            if det1_result[i]["file_name"] == gt["file_name"]:
                det1 = det1_result[i]
            if det2_result[i]["file_name"] == gt["file_name"]:
                det2 = det2_result[i]
            else:
                try:
                    det1_idx = det1_file_names.index(gt["file_name"])
                    det1 = det1_result[det1_idx]
                    det2_idx = det2_file_names.index(gt["file_name"])
                    det2 = det2_result[det2_idx]
                except ValueError:
                    continue
            yield gt, det1, det2

def _check_vo_pair_in_image(img_anno, vb_id, obj_id):
    r"""check if a vb-obj pair exists in one image,
    based on `img_anno`.
    """
    obj_key, vb_key = None, None
    if "annotations" in img_anno:
        obj_key, vb_key = "annotations", "hoi_annotation"
    else:
        obj_key, vb_key = "predictions", "hoi_prediction"
    objs = img_anno[obj_key]
    for hoi in img_anno[vb_key]:
        if hoi["category_id"] == vb_id and objs[hoi["object_id"]]["category_id"] == obj_id:
            return True
    return False

def create_classwise_generator_by_gt(gt_result, det_result, vb_id=None, obj_id=None):
    r"""create a generator which yields paired gt and det for a specific verb category `vb_id`,
    according to the labels of `gt_result`.

    Args:
        vb_id (int): verb category id, for HICO-DET, should be [1, 117]
        obj_id (int): object category id, for HICO-DET, should be [1, 90]
    """
    assert (vb_id is not None or obj_id is not None), "Either provide verb id or object id."
    det_file_names = [det["file_name"] for det in det_result]
    for i, gt in enumerate(gt_result):
        if vb_id is not None and obj_id is not None:
            # select images that contains vb-obj pair, not vb and obj separately
            if not _check_vo_pair_in_image(gt, vb_id, obj_id):
                continue
        elif vb_id is not None:
            gt_category_ids = set([hoi["category_id"] for hoi in gt["hoi_annotation"]])
            if vb_id not in gt_category_ids:
                continue
        elif obj_id is not None:
            gt_category_ids = set([box["category_id"] for box in gt["annotations"]])
            if obj_id not in gt_category_ids:
                continue
        # got gt, now get paired det
        if det_result[i]["file_name"] == gt["file_name"]:
            det = det_result[i]
        else:
            try:
                det_idx = det_file_names.index(gt["file_name"])
                det = det_result[det_idx]
            except ValueError:
                continue
        yield gt, det

def create_classwise_generator_by_det(gt_result, det_result, vb_id=None, obj_id=None):
    r"""create a generator which yields paired gt and det for a specific verb category `vb_id`,
    according to the labels of `det_result`.

    Args:
        vb_id (int): verb category id, for HICO-DET, should be [1, 117]
        obj_id (int): object category id, for HICO-DET, should be [1, 90]
    """
    assert (vb_id or obj_id), "Either provide verb id or object id."
    gt_file_names = [gt["file_name"] for gt in gt_result]
    for i, det in enumerate(det_result):
        if vb_id is not None and obj_id is not None:
            # select images that contains vb-obj pair, not vb and obj separately
            if not _check_vo_pair_in_image(det, vb_id, obj_id):
                continue
        elif vb_id is not None:
            det_category_ids = set([hoi["category_id"] for hoi in det["hoi_prediction"]])
            if vb_id not in det_category_ids:
                continue
        elif obj_id is not None:
            det_category_ids = set([box["category_id"] for box in det["predictions"]])
            if obj_id not in det_category_ids:
                continue
        # got det, now get paired gt
        if gt_result[i]["file_name"] == det["file_name"]:
            gt = gt_result[i]
        else:
            try:
                gt_idx = gt_file_names.index(det["file_name"])
                gt = gt_result[gt_idx]
            except ValueError:
                continue
        yield gt, det

def keep_det_with_gt(gt_result, det_result):
    r"""Keep only det for images with gt
    Return:
        a list of dict which is a new det_result
    """
    gt_file_names = [gt["file_name"] for gt in gt_result]
    return [det for det in det_result if det["file_name"] in gt_file_names]

#####################################################################
# -------------------------- Draw Utils --------------------------- #
#####################################################################

def draw_rec(bbox, color='cyan', linewidth=2):
    r'''Draw rectangle on image based on bbox coordinates

    Args:
        bbox: list, [x0, y0, x1, y1]
        color: the box color
    '''
    x0, y0, x1, y1 = bbox
    rect = patches.Rectangle((x0, y0),
            x1-x0,
            y1-y0,
            linewidth=linewidth,
            edgecolor=color,
            fill = False)
    return rect


#####################################################################
# -------------------- MultiProcessing Utils ---------------------- #
#####################################################################

def reorganize_func_1st(start, end, func, dataset, *args):
    r'''Process dataset with func from start to end.

    Args:
        start: int, the start index of dataset
        end: int, the end index of dataset
        func: the processing function
        dataset: list, the dataset to be processed
        *args: the args to be passed to `func`,
            after other specified arguments
    '''
    for i in tqdm(range(start, end)):
        func(dataset[i], *args)
    return

def reorganize_func_2nd(start, end, func, dataset, *args):
    r'''Process dataset with func from start to end,
        with 1st of `args` treated specially.

    Args:
        start: int, the start index of dataset
        end: int, the end index of dataset
        func: the processing function
        dataset: list, the dataset to be processed
        *args: the args to be passed to `func`,
            after other specified arguments
    '''
    for i in tqdm(range(start, end)):
        func(dataset[i], args[0][i], *(args[1:]))
    return

def multi_process_image(func, dataset, target, NUM_PROCESS, *args):
    r'''Multi process with func

    Args:
        dataset: list of instances, the dataset to be processed
        func: function to process one instance in the dataset
        target: function to perform "func" one a list of instances
        NUM_PROCESS: number of process to use
    '''
    gap = len(dataset) // NUM_PROCESS
    # TODO: fix gap too small waste
    start_time = time.time()
    print("Current parent process: {}".format(os.getpid()))
    ps = []
    for i in range(NUM_PROCESS):
        start = i * gap
        end = min((i + 1) * gap, len(dataset) - 1)
        print('Start child process {} for instances [{}, {})'.format(i, start, end))
        ps.append(Process(target=target, args=(start, end, func, dataset, *args)))
    if end < len(dataset):
        print('Start child process {} for instances [{}, {})'.format(i + 1, end, len(dataset)))
        ps.append(Process(target=target, args=(end, len(dataset), func, dataset, *args)))
    print("Waiting for all child process to finish.")
    for pi in ps:
        pi.start()
    for pi in ps:
        pi.join()
    end_time = time.time()
    print("Total time cost {} seconds".format((end_time - start_time)))

#####################################################################
# ------------------------- Other Utils --------------------------- #
#####################################################################

def get_name(file_path):
    r"""Get file name without extension from file path.
    """
    return file_path.split("/")[-1].split(".")[-2]

def get_file_path_from(old_path, file_name):
    r"""Given a file path, substitute the file name with a new one

    Args:
        old_path (str): old file path
        file_name (str): the new file name
    Return:
        a str representing the new file path
    """
    return "/".join(old_path.split("/")[:-1] + [file_name])

def mkdir_multiprocess(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass
