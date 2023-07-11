#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   eval_detection.py
@Time    :   2020/09/01 17:59:14
@Author  :   HuYue
@Contact :   huyue@megvii.com
@Desc    :   evaluate the detection result of ppdm
'''

# this script is used for the calculation of detection mAP on HOI dataset

import numpy as np
from loguru import logger
from collections import OrderedDict
from tqdm import tqdm

from utils import (load_hico_category, list2dic, get_iou_np,
load_res_from_json, create_gt_det_generator, keep_det_with_gt)
from argparser import args

class mAPEval:
    def __init__(self, dt_path, gt_path, hico_list_dir):
        r"""Evaluating detection mAP.
        Args:
            dt_path (str): path of detection result json file
            gt_path (str): path of gt json file provided by the dataset
            hico_list_dir (str): dir where hico-list txt file exists
        """
        self.dt_path = dt_path
        self.gt_path = gt_path

        self.dt_list = list()
        self.gt_list = list()

        _, obj_c, _ = load_hico_category(hico_list_dir)
        self.obj_name2id, self.obj_id2name = list2dic(obj_c)

        self.load_anno()
        assert len(self.dt_list) == len(self.gt_list)
        self.nr_eval_img = len(self.gt_list)
        self.eval = {}

        self.tp = OrderedDict()
        self.fp = OrderedDict()
        self.score = OrderedDict()
        self.sum_gt = OrderedDict()



    def nms(self, dets, thresh=0.5):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, -1]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order.item(0)
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def merge_dt_gt(self, dt, gt, key_box="bbox", key_score="score"):
        dt_box = []
        gt_box = []
        # sub_thres = 0.1
        obj_thres = 0.0

        for item in dt["predictions"]:
            if float(item[key_score]) < obj_thres:
                continue
            # if item[key_score] < sub_thres and item["category_id"] == 1:
            #     continue
            if key_box in item:
                if 'category_id' in item:
                    dt_box.append(item[key_box] + [item['category_id']] + [float(item[key_score])])
                else:
                    dt_box.append(item[key_box] + [self.obj_name2id[item['category']]] + [item[key_score]])

        for item in gt["annotations"]:
            if item.get(key_box) is not None:
                gt_box.append(item[key_box] + [item['category_id']] + [1.0])

        list2nparr2d = lambda x, y: np.array(x, dtype="float32") if len(x) > 0 else np.zeros((0, y), dtype="float32")

        self.dt_list.append(list2nparr2d(dt_box, 6))
        self.gt_list.append(list2nparr2d(gt_box, 6))

    def load_anno(self):
        # load gt
        gt_result = load_res_from_json(self.gt_path)
        # load det
        det_result = load_res_from_json(self.dt_path)

        if len(gt_result) != len(det_result):
            print("[Warning] prediction has {} images while gt has {}".format(len(det_result), len(gt_result)))
        det_result = keep_det_with_gt(gt_result, det_result)

        gt_det_generator = create_gt_det_generator(
            gt_result, det_result, (0, len(gt_result))
        )
        for gt, det in gt_det_generator:
            self.merge_dt_gt(det, gt)

    # def voc_ap(self, rec, prec):
    #     r"""VOC07 AP metric. see https://github.com/dmlc/gluon-cv/blob/master/gluoncv/utils/metrics/voc_detection.py#L273
    #     """
    #     ap = 0.
    #     for t in np.arange(0., 1.1, 0.1):
    #         if np.sum(rec >= t) == 0:
    #             p = 0
    #         else:
    #             p = np.max(prec[rec >= t])
    #         ap = ap + p / 11.
    #     return ap

    def voc_ap(self, rec, prec):
        r"""VOC12 AP metric. see https://github.com/dmlc/gluon-cv/blob/master/gluoncv/utils/metrics/voc_detection.py#L226
        """
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def compute_tpfp(self, match_iou=0.5, nms_iou=0.5):
        tp = OrderedDict()
        fp = OrderedDict()
        score = OrderedDict()
        sum_gt = OrderedDict()

        for img_idx in tqdm(range(self.nr_eval_img), "Calculating TP/FP..."):
            dt = self.dt_list[img_idx]
            gt = self.gt_list[img_idx]
            uni_cls = np.unique(dt[:, 4])
            gt_uni_cls = np.unique(gt[:, 4])

            for cls_idx in uni_cls:
                cls_idx = int(cls_idx)
                if cls_idx not in tp:
                    tp[cls_idx] = []
                if cls_idx not in fp:
                    fp[cls_idx] = []
                if cls_idx not in score:
                    score[cls_idx] = []

                bbox_idx = np.where(dt[:, 4]==cls_idx)[0]
                cur_dt = dt.copy()[bbox_idx]
                cur_keep = self.nms(cur_dt, nms_iou)
                # if len(cur_keep) != len(cur_dt):
                #     print('{} / {}'.format(len(cur_keep), len(cur_dt)))
                cur_dt = cur_dt[cur_keep]
                cur_dt = cur_dt[np.argsort(-cur_dt[:, -1])]

                gt_bbox_idx = np.where(gt[:, 4]==cls_idx)[0]
                if len(gt_bbox_idx) < 1:
                    continue
                cur_gt = gt.copy()[gt_bbox_idx]


                dt_xy = cur_dt.copy()
                # dt_xy[:, 2:4] = dt_xy[:, 0:2] + dt_xy[:, 2:4]

                gt_xy = cur_gt.copy()
                # gt_xy[:, 2:4] = gt_xy[:, 0:2] + gt_xy[:, 2:4]

                iou = get_iou_np(dt_xy[:, :4], gt_xy)

                nr_dt, nr_gt = iou.shape
                groundtruths = list(range(nr_gt))
                for i in range(nr_dt):
                    if len(groundtruths) == 0:
                        break
                    j = max(groundtruths, key=lambda j: iou[i, j])
                    if iou[i, j] > match_iou:
                        groundtruths.remove(j)
                        fp[cls_idx].append(0)
                        tp[cls_idx].append(1)
                    else:
                        fp[cls_idx].append(1)
                        tp[cls_idx].append(0)

                    score[cls_idx].append(dt_xy[i, -1])

            for cls_idx in gt_uni_cls:
                if cls_idx not in sum_gt:
                    sum_gt[cls_idx] = 0
                gt_bbox_idx = np.where(gt[:, 4]==cls_idx)[0]
                nr_gt = len(gt_bbox_idx)
                sum_gt[cls_idx] += nr_gt

        result = {
            "match_iou": match_iou,
            "nms_iou": nms_iou,
            "tp": tp,
            "fp": fp,
            "score": score,
            "sum_gt": sum_gt
        }
        return result
    
    def compute_map(self, tpfp):
        score = tpfp["score"]
        sum_gt = tpfp["sum_gt"]
        tp = tpfp["tp"]
        fp = tpfp["fp"]
        match_iou = tpfp["match_iou"]

        num_class = len(score)
        ap = np.zeros(num_class)
        max_recall = np.zeros(num_class)
        inds = list(sorted(score.keys()))

        # logger.info("|-------- Class -------|----- ap -----|----- max recall ----|")
        for i in range(num_class):
            ind = int(inds[i])
            cur_sum_gt = sum_gt[ind]
            if cur_sum_gt == 0:
                continue
            cur_tp = np.asarray((tp[ind]).copy())
            cur_fp = np.asarray((fp[ind]).copy())
            res_num = len(cur_tp)
            if res_num == 0:
                continue
            cur_score = np.asarray(score[ind].copy())
            sort_inds = np.argsort(-cur_score)
            cur_fp = cur_fp[sort_inds]
            cur_tp = cur_tp[sort_inds]
            cur_fp = np.cumsum(cur_fp)
            cur_tp = np.cumsum(cur_tp)
            rec = cur_tp / cur_sum_gt
            prec = cur_tp / (cur_fp + cur_tp)
            ap[i] = self.voc_ap(rec, prec)
            max_recall[i] = np.max(rec)

            logger.info("|\t{:10}\t|\t{:.04f} \t|\t{:.04f}\t|".format(self.obj_id2name[ind], ap[i], max_recall[i]))

        # logger.info("|----------------------------------------------------------|\n")

        mAP = np.mean(ap[:])
        m_rec = np.mean(max_recall[:])
        logger.info("|------- MatchIou ------|---------- mAP --------|---- max recall  ------|")
        logger.info("|\t {:.04f} \t|\t {:.04f} \t|\t{:.04f}  \t|".format(match_iou, mAP, m_rec))
        logger.info("|-----------------------------------------------------------------------|\n")

        # save AP
        self.ap = {}
        for i, ap_val in enumerate(ap):
            self.ap[int(inds[i])] = ap_val
        return mAP, m_rec

    def get_ap(self):
        r"""return AP as an (unordered) dict,
        where k is one of 80 object index,
        and v is the corresponding AP.
        """
        return self.ap

def get_mAP_acrossIOU_eval(dt_path, gt_path, hico_list_dir):
    """

    """
    map_eval = mAPEval(dt_path, gt_path, hico_list_dir)
    """
        Calculate mAP averaged over IOU=0.5~1.0.
    """
    mAPs = []
    for match_iou in np.arange(0.5, 1.0, 0.05):
        tpfp = map_eval.compute_tpfp(match_iou, nms_iou=1.0)
        mAP, _ = map_eval.compute_map(tpfp)
        mAPs.append(mAP)
    logger.info("mAP: {}".format(np.mean(mAPs)))

def get_mAP_eval(dt_path, gt_path, hico_list_dir, match_iou=0.5):
    """
        Calculate mAP @`match_iou`.
    """
    map_eval = mAPEval(dt_path, gt_path, hico_list_dir)
    tpfp = map_eval.compute_tpfp(match_iou, nms_iou=1.0)
    _, _ = map_eval.compute_map(tpfp)

def get_AP_dict(dt_path, gt_path, hico_list_dir, match_iou=0.5):
    r"""Return AP @`match_iou` for each category.
    """
    map_eval = mAPEval(dt_path, gt_path, hico_list_dir)
    tpfp = map_eval.compute_tpfp(match_iou, nms_iou=1.0)
    _, _ = map_eval.compute_map(tpfp)
    return map_eval.get_ap()

if __name__ == '__main__':
    print("Starting object detection result mAP evaluation")
    gt_path = args.gt_path
    dt_path = args.det_path
    get_mAP_eval(dt_path, gt_path, args.hico_list_dir)
