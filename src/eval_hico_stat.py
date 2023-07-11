#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# evaluate model AP, max_recall & mAP.
# also supports AP & max_recall between 2 models
# also supports error diagnosis

# code based on
# https://github.com/YueLiao/PPDM/blob/master/src/lib/eval/hico_eval.py

from functools import partial
import os
import csv

import numpy as np
from numpy.lib.npyio import savez_compressed
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_res_from_json, load_hico_category, get_file_path_from
from argparser import args

class HICO():
    def __init__(self, annotation_file, hico_list_dir):
        self.annotations = load_res_from_json(annotation_file)
        self.train_annotations = load_res_from_json(get_file_path_from(annotation_file, 'trainval_hico.json'))
        self.category_info = load_hico_category(hico_list_dir)
        self.overlap_iou = 0.5
        self.verb_name_dict = []
        self.fp = {}
        self.tp = {}
        self.score = {}
        self.fnames = {}
        self.sum_gt = {}
        self.image_names_for_categories = {}
        self.file_name = []
        self.train_sum = {}
        for gt_i in self.annotations:
            self.file_name.append(gt_i['file_name'])
            gt_hoi = gt_i['hoi_annotation']
            gt_bbox = gt_i['annotations']
            for gt_hoi_i in gt_hoi:
                if isinstance(gt_hoi_i['category_id'], str):
                    gt_hoi_i['category_id'] = int(gt_hoi_i['category_id'].replace('\n', ''))
                triplet = [gt_bbox[gt_hoi_i['subject_id']]['category_id'],gt_bbox[gt_hoi_i['object_id']]['category_id'],gt_hoi_i['category_id']]
                if triplet not in self.verb_name_dict:
                    self.verb_name_dict.append(triplet)
                if self.verb_name_dict.index(triplet) not in self.sum_gt.keys():
                    self.sum_gt[self.verb_name_dict.index(triplet)] =0
                    self.image_names_for_categories[self.verb_name_dict.index(triplet)] = set()
                self.sum_gt[self.verb_name_dict.index(triplet)] += 1
                self.image_names_for_categories[self.verb_name_dict.index(triplet)].add(gt_i["file_name"])
        for train_i in self.train_annotations:
            train_hoi = train_i['hoi_annotation']
            train_bbox = train_i['annotations']
            for train_hoi_i in train_hoi:
                if isinstance(train_hoi_i['category_id'], str):
                    train_hoi_i['category_id'] = int(train_hoi_i['category_id'].replace('\n', ''))
                triplet = [train_bbox[train_hoi_i['subject_id']]['category_id'],train_bbox[train_hoi_i['object_id']]['category_id'],train_hoi_i['category_id']]
                if triplet not in self.verb_name_dict:
                    continue
                if self.verb_name_dict.index(triplet) not in self.train_sum.keys():
                    self.train_sum[self.verb_name_dict.index(triplet)] = 0
                self.train_sum[self.verb_name_dict.index(triplet)] += 1
        for i in range(len(self.verb_name_dict)):
            self.fp[i] = []
            self.tp[i] = []
            self.score[i] = []
            self.fnames[i] = []
        self.r_inds = []
        self.c_inds = []
        for id in self.train_sum.keys():
            if self.train_sum[id] < 10:
                self.r_inds.append(id)
            else:
                self.c_inds.append(id)
        self.int_inds = []
        self.non_inds = []
        hoi_c, obj_c, vb_c = self.category_info
        for i in range(len(self.verb_name_dict)):
            vb_name = vb_c[self.verb_name_dict[i][2] - 1]
            if vb_name == "no_interaction":
                self.non_inds.append(i)
            else:
                self.int_inds.append(i)
        self.num_class = len(self.verb_name_dict)

    def evalution(self, predict_annot, verbose=True, demo_dir=None):
        r'''
        predict_annot: same struct as the gt_annotations
        predictions --> annotations [list of dict]
        {
            'bbox': [x0, y0, x1, y1]
            'category_id': 
        }
        hoi_prediction --> hoi_annotation [list of dict]
        {
            'subject_id':
            'object_id':
            'category_id':
        }
        '''
        for pred_i in tqdm(predict_annot, "Calculating TP/FP..."):
            if pred_i['file_name'] not in self.file_name:
                continue
            gt_i = self.annotations[self.file_name.index(pred_i['file_name'])]
            gt_bbox = gt_i['annotations']
            if len(gt_bbox)!=0 and len(pred_i['hoi_prediction'])!=0:
                pred_bbox = pred_i['predictions']  # already converted zero-based to one-based indices
                try:
                    bbox_pairs, bbox_ov = self.compute_iou_mat(gt_bbox, pred_bbox)
                except:
                    import ipdb; ipdb.set_trace()
                pred_hoi = pred_i['hoi_prediction']
                gt_hoi = gt_i['hoi_annotation']
                self.compute_fptp(pred_hoi, gt_hoi, bbox_pairs, pred_bbox,bbox_ov, pred_i['file_name'])
                # self.compute_fptp_otherwise(pred_hoi, gt_hoi, bbox_pairs, pred_bbox, gt_bbox, bbox_ov)
            else:
                pred_bbox = pred_i['predictions']  # already converted zero-based to one-based indices
                for i, pred_hoi_i in enumerate(pred_i['hoi_prediction']):
                    triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'],
                               pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                    verb_id = self.verb_name_dict.index(triplet)
                    self.tp[verb_id].append(0)
                    self.fp[verb_id].append(1)
                    self.score[verb_id].append(pred_hoi_i['score'])
        mAP = self.compute_map(verbose=verbose, demo_dir=demo_dir)
        return mAP

    def evalution_known_object(self, predict_annot, verbose=True, demo_dir=None):
        for pred_i in tqdm(predict_annot, "Calculating TP/FP..."):
            if pred_i['file_name'] not in self.file_name:
                continue
            gt_i = self.annotations[self.file_name.index(pred_i['file_name'])]
            gt_bbox = gt_i['annotations']
            if len(gt_bbox)!=0 and len(pred_i['hoi_prediction'])!=0:
                pred_bbox = pred_i['predictions']  # already converted zero-based to one-based indices
                try:
                    bbox_pairs, bbox_ov = self.compute_iou_mat(gt_bbox, pred_bbox)
                except:
                    import ipdb; ipdb.set_trace()
                pred_hoi = pred_i['hoi_prediction']
                gt_hoi = gt_i['hoi_annotation']
                self.compute_fptp_known_object(pred_hoi, gt_hoi, bbox_pairs, pred_bbox, gt_bbox, bbox_ov)
            else:
                pred_bbox = pred_i['predictions']  # already converted zero-based to one-based indices
                for i, pred_hoi_i in enumerate(pred_i['hoi_prediction']):
                    triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'],
                               pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                    verb_id = self.verb_name_dict.index(triplet)
                    self.tp[verb_id].append(0)
                    self.fp[verb_id].append(1)
                    self.score[verb_id].append(pred_hoi_i['score'])
        mAP = self.compute_map(verbose=verbose, demo_dir=demo_dir)
        return mAP

    def compute_map(self, verbose=True, demo_dir=None):
        r'''Compute AP and mAP values.

        Args:
            verbose (bool): whether or not to print AP values.
        .. note::
            self.verb_name_dict (length 600): dict of triplet, key is the hoi id, value is triplet <h-i-o>
            self.sum_gt (length 600): dict of count, key is the hoi id, value is the hoi gt amount
            self.tp / self.fp (length 600): dict of list, the key is the hoi id, value is the predictions (true 1 or false 0)
            self.score: the same structure as self.tp and self.fp, the value is the prediction score(sub_score*obj_score*rel_score) (between 0 - 1)
        '''
        hoi_c, obj_c, vb_c = self.category_info
        ap = np.zeros(self.num_class)
        max_recall = np.zeros(self.num_class)
        self.pres = []
        self.recs = []
        for i in range(len(self.verb_name_dict)):
            sum_gt = self.sum_gt[i]
            if sum_gt == 0:
                continue
            tp = np.asarray((self.tp[i]).copy())
            fp = np.asarray((self.fp[i]).copy())
            res_num = len(tp)
            if res_num == 0:
                continue
            score = np.asarray(self.score[i].copy())
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gt
            prec = tp / (fp + tp)
            ap[i] = self.voc_ap(rec, prec)
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))
            self.pres.append(mpre)
            self.recs.append(mrec)
            # if i == 10:
            #     import ipdb;ipdb.set_trace()
            max_recall[i] = np.max(rec)
            # if verbose:
            #     print('class {} {:10}-{:10} | --- ap: {:.04f} --- | --- max recall: {:.04f} ---|'.format(i, vb_c[self.verb_name_dict[i][2]-1], obj_c[self.verb_name_dict[i][1]-1], ap[i], max_recall[i]))
        self.ap = ap[:]
        self.max_recall = max_recall[:]
        mAP = np.mean(ap[:])
        mAP_rare = np.mean(ap[self.r_inds])
        mAP_nonrare = np.mean(ap[self.c_inds])
        mAP_int = np.mean(ap[self.int_inds])
        mAP_non = np.mean(ap[self.non_inds])
        m_rec = np.mean(max_recall[:])
        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP nonrare: {}  max recall: {} mAP interaction: {} mAP no_interaction: {}'.format(mAP, mAP_rare, mAP_nonrare, m_rec, mAP_int, mAP_non))
        print('--------------------')

        return mAP

    def voc_ap(self, rec, prec):
        r"""Calculate AP in VOC 2007 style.

        Args:
            rec (list of float): list of recall values
            prec (list of float): list of precision values
        Return:
            A float representing AP value
        """
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    # def voc_ap(self, rec, prec):
    #     r"""VOC12 AP metric. see https://github.com/dmlc/gluon-cv/blob/master/gluoncv/utils/metrics/voc_detection.py#L226
    #     """
    #     mrec = np.concatenate(([0.], rec, [1.]))
    #     mpre = np.concatenate(([0.], prec, [0.]))
    #     for i in range(mpre.size - 1, 0, -1):
    #         mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    #     i = np.where(mrec[1:] != mrec[:-1])[0]
    #     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    #     return ap

    def compute_fptp(self, pred_hoi, gt_hoi, match_pairs, pred_bbox, bbox_ov, file_name):
        r"""Generate FP/TP lists for one image using `pred_hoi` and `gt_hoi`.

        Args:
            pred_hoi (dict): pred["hoi_prediction"]
            gt_hoi (dict): gt["hoi_annotation"]
            match_pairs (dict): {pred_id: [list of matched gt id]}
            pred_bbox (list of dict): pred["predictions"]
            bbox_ov (dict): {pred_id: [list of IOU value with matched gt]}
        """
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                is_match = 0
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
                    # for each predicted hoi, if there are two gt matched,
                    # pick the gt with the largest object score
                    for gt_id in range(len(gt_hoi)):
                    # for gt_id in np.where(vis_tag==0)[0]:
                        gt_hoi_i = gt_hoi[gt_id]
                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            min_ov_gt = min(pred_sub_ov[pred_sub_ids.index(gt_hoi_i['subject_id'])], pred_obj_ov[pred_obj_ids.index(gt_hoi_i['object_id'])])
                            if min_ov_gt > max_ov:
                                max_ov = min_ov_gt
                                max_gt_id = gt_id
                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                try:
                    triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'], pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                except:
                    import ipdb; ipdb.set_trace()
                if triplet not in self.verb_name_dict:
                    continue
                verb_id = self.verb_name_dict.index(triplet)
                if is_match == 1 and vis_tag[max_gt_id] == 0:
                    self.fp[verb_id].append(0)
                    self.tp[verb_id].append(1)
                    vis_tag[max_gt_id] = 1
                else:
                    self.fp[verb_id].append(1)
                    self.tp[verb_id].append(0)
                self.score[verb_id].append(pred_hoi_i['score'])
                self.fnames[verb_id].append(file_name)

    def compute_fptp_known_object(self, pred_hoi, gt_hoi, match_pairs, pred_bbox, gt_bbox, bbox_ov):
        r"""Generate FP/TP lists for one image using `pred_hoi` and `gt_hoi`.
        This is done in the mode of ws-vrd evaluation.

        Args:
            pred_hoi (dict): pred["hoi_prediction"]
            gt_hoi (dict): gt["hoi_annotation"]
            match_pairs (dict): {pred_id: [list of matched gt id]}
            pred_bbox (list of dict): pred["predictions"]
            bbox_ov (dict): {pred_id: [list of IOU value with matched gt]}
        """
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        gt_obj_sets = set([box["category_id"] for box in gt_bbox])
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                # ===== known vs. default diff =====
                # for each hoi class, eval only performed on the images containing that gt obj
                pred_obj = pred_bbox[pred_hoi_i["object_id"]]["category_id"]
                if pred_obj not in gt_obj_sets:
                    continue
                # ===== known vs. default diff =====
                is_match = 0
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
                    # for each predicted hoi, if there are two gt matched,
                    # pick the gt with the largest object score
                    for gt_id in range(len(gt_hoi)):
                    # for gt_id in np.where(vis_tag==0)[0]:
                        gt_hoi_i = gt_hoi[gt_id]
                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            min_ov_gt = min(pred_sub_ov[pred_sub_ids.index(gt_hoi_i['subject_id'])], pred_obj_ov[pred_obj_ids.index(gt_hoi_i['object_id'])])
                            if min_ov_gt > max_ov:
                                max_ov = min_ov_gt
                                max_gt_id = gt_id
                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                try:
                    triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'], pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                except:
                    import ipdb; ipdb.set_trace()
                if triplet not in self.verb_name_dict:
                    continue
                verb_id = self.verb_name_dict.index(triplet)
                if is_match == 1 and vis_tag[max_gt_id] == 0:
                    self.fp[verb_id].append(0)
                    self.tp[verb_id].append(1)
                    vis_tag[max_gt_id] = 1
                else:
                    self.fp[verb_id].append(1)
                    self.tp[verb_id].append(0)
                self.score[verb_id].append(pred_hoi_i['score'])


    def compute_fptp_otherwise(self, pred_hoi, gt_hoi, match_pairs, pred_bbox, gt_bbox, bbox_ov):
        r"""Generate FP/TP lists for one image using `pred_hoi` and `gt_hoi`.
        This is done in the mode of ws-vrd evaluation.

        Args:
            pred_hoi (dict): pred["hoi_prediction"]
            gt_hoi (dict): gt["hoi_annotation"]
            match_pairs (dict): {pred_id: [list of matched gt id]}
            pred_bbox (list of dict): pred["predictions"]
            bbox_ov (dict): {pred_id: [list of IOU value with matched gt]}
        """
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        gt_vo_sets = set([(gt_hoi_i["category_id"], gt_bbox[gt_hoi_i["object_id"]]["category_id"]) for gt_hoi_i in gt_hoi])
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                # ===== ws-vrd diff =====
                # for each hoi class, eval only performed on the images containing that gt class
                pred_vo = (pred_hoi_i["category_id"], pred_bbox[pred_hoi_i["object_id"]]["category_id"])
                if pred_vo not in gt_vo_sets:
                    continue
                # ===== ws-vrd diff =====
                is_match = 0
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
                    # for each predicted hoi, if there are two gt matched,
                    # pick the gt with the largest object score
                    for gt_id in range(len(gt_hoi)):
                    # for gt_id in np.where(vis_tag==0)[0]:
                        gt_hoi_i = gt_hoi[gt_id]
                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            min_ov_gt = min(pred_sub_ov[pred_sub_ids.index(gt_hoi_i['subject_id'])], pred_obj_ov[pred_obj_ids.index(gt_hoi_i['object_id'])])
                            if min_ov_gt > max_ov:
                                max_ov = min_ov_gt
                                max_gt_id = gt_id
                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                try:
                    triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'], pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                except:
                    import ipdb; ipdb.set_trace()
                if triplet not in self.verb_name_dict:
                    continue
                verb_id = self.verb_name_dict.index(triplet)
                if is_match == 1 and vis_tag[max_gt_id] == 0:
                    self.fp[verb_id].append(0)
                    self.tp[verb_id].append(1)
                    vis_tag[max_gt_id] = 1
                else:
                    self.fp[verb_id].append(1)
                    self.tp[verb_id].append(0)
                self.score[verb_id].append(pred_hoi_i['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2, thres=0.5):
        r'''Match a gt bbox for each detected bbox (iou >= `thres`).

        Args:
            bbox_list1 (list of dict): gt["annotationss"]
            bbox_list2 (list of dict): pred["predictions]
            thres (optional, float): threshold for determining box overlapping.
            default value: 0.5.
        Return:
            a dict of {pred_id: [list of matched gt id]}
            a dict of {pred_id: [list of IOU value with matched gt]}
        '''
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}, {}
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov = iou_mat.copy()
        iou_mat[iou_mat >= thres] = 1
        iou_mat[iou_mat < thres] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pairs_ov={}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pairs_ov[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pairs_ov[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict,match_pairs_ov

    def compute_IOU(self, bbox1, bbox2):
        r"""Calculating IOU values for two box `bbox1` and `bbox2`.
        .. note::
            If `bbox1` and `bbox2` has different category_id, 0 will be returned. 
        Args:
            bbox1 (dict): dict with keys "category_id" and "bbox"
            bbox2 (dict): dict with keys "category_id" and "bbox"
        Return:
            A float value in [0, 1] indicating the IOU value of `bbox1` and `bbox2`.
            0 if different they has different category.
        """
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
            S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
                return intersect / (sum_area - intersect)
        else:
            return 0

    def add_One(self,prediction):  #Add 1 to all coordinates
        for i, pred_bbox in enumerate(prediction):
            rec = pred_bbox['bbox']
            rec[0]+=1
            rec[1]+=1
            rec[2]+=1
            rec[3]+=1
        return prediction

    def get_ap(self):
        r"""get a numpy array of AP values for 600 categories,
        and the indices corresponds to `self.verb_name_dict`.
        """
        return self.ap[:]

    def get_max_recall(self):
        r"""get a numpy array of max_recall values for 600 categories,
        and the indices corresponds to `self.verb_name_dict`.
        """
        return self.max_recall[:]

    def get_verb_name_dict(self):
        r"""get a list of [1, obj_id (1~90), vb_id (1~117)],
        whose length is 600.
        """
        return self.verb_name_dict

    def get_sum_gt(self):
        r"""get count num of HOI pairs for each category,
        from the test set.
        return: dict of {verb_name_dict_id: test_count_int}
        """
        return self.sum_gt

    def get_train_sum_gt(self):
        r"""get count num of HOI pairs for each category,
        from the train set.
        return: dict of {verb_name_dict_id: train_count_int}
        """
        return self.train_sum

    def get_rare_nonrare_inds(self):
        return self.r_inds, self.c_inds

    def reset(self):
        r"""Reset values for the evaluation of another model.
        """
        self.fp = {}
        self.tp = {}
        self.score = {}
        self.fnames = {}
        for i in range(len(self.verb_name_dict)):
            self.fp[i] = []
            self.tp[i] = []
            self.score[i] = []
            self.fnames[i] = []

    def diagnosis(self, predict_annot, demo_dir=None):
        r'''
        predict_annot: same struct as the gt_annotations
        predictions --> annotations [list of dict]
        {
            'bbox': [x0, y0, x1, y1]
            'category_id':
        }
        hoi_prediction --> hoi_annotation [list of dict]
        {
            'subject_id':
            'object_id':
            'category_id':
        }
        '''
        self.tp = [[] for i in range(self.num_class)]
        self.fp1 = [[] for i in range(self.num_class)]  # bck
        self.fp2 = [[] for i in range(self.num_class)]  # person misloc
        self.fp3 = [[] for i in range(self.num_class)]  # mis-grouping
        self.fp4 = [[] for i in range(self.num_class)]  # obj misloc
        self.fp5 = [[] for i in range(self.num_class)]  # incorrect obj label
        self.fp6 = [[] for i in range(self.num_class)]  # incorrect action label
        self.fp7 = [[] for i in range(self.num_class)]  # duplicate pairs
        self.diag_score = [[] for i in range(self.num_class)]

        for pred_i in tqdm(predict_annot, "Dianosing error..."):
            if pred_i['file_name'] not in self.file_name:
                continue
            gt_i = self.annotations[self.file_name.index(pred_i['file_name'])]
            gt_bbox = gt_i['annotations']
            pred_bbox = pred_i['predictions']  # alrady converted zero-based to one-based indices
            try:
                bbox_pairs, bbox_ov = self.compute_iou_mat(gt_bbox, pred_bbox)
                bbox_pairs_low_thres, bbox_ov_low_thres = self.compute_iou_mat(gt_bbox, pred_bbox, thres=0.1)
            except:
                import ipdb; ipdb.set_trace()
            pred_hoi = pred_i['hoi_prediction']
            gt_hoi = gt_i['hoi_annotation']
            self.diagnose_img_error(pred_hoi, gt_hoi, bbox_pairs, bbox_pairs_low_thres, pred_bbox, bbox_ov, bbox_ov_low_thres)
        self.compute_diag_result(demo_dir=demo_dir)

    def compute_diag_result(self, demo_dir=None):
        r"""compute error diagnosis results.
        """
        hoi_c, obj_c, vb_c = self.category_info
        fp_types = ["person bck", "person misloc", "obj bck",
            "obj misloc", "incorrect obj label", "incorrect interaction",
            "duplicate"
        ]
        fp_counts = [0] * 7
        fp_frac_sum = [0 for i in range(7)]
        fp_cls_count = 0
        fp_frac_matrix = np.zeros((self.num_class, 7))
        print_template = "for verb_id {} ({:5.5}-{:10.10}), num_gt {}, num_pred {}, recall {:.04f}. " \
            + "Error: bck {:.04f}, person_misloc {:.04f}, mis_grouping {:.04f}, obj_misloc {:.04f}, " \
            + "incorrect_obj_label {:.04f}, incorrect_action_label {:.04f}, duplicate {:.04f}."
        # calculate score
        result_seq = []
        tp_sum = 0
        for i in range(self.num_class):
            score = np.array(self.diag_score[i], dtype=np.float32)
            score_idx = score.argsort()[::-1]
            # num_inst = min(self.sum_gt[i], len(self.tp[i]))
            num_inst = len(self.tp[i])
            # if sum(self.tp[i]) > 0:
            #     num_inst = len(self.tp[i]) - self.tp[i][::-1].index(1) - 1
            # else:
            #     num_inst = 0
            tp_all = np.array(self.tp[i], dtype=np.float32)[score_idx]
            tp = tp_all[:num_inst]
            tp_sum += int(sum(tp))
            vb_name = vb_c[self.verb_name_dict[i][2] - 1]
            obj_name = obj_c[self.verb_name_dict[i][1] - 1]
            if num_inst - np.sum(tp) == 0:
                result_seq.append([i, vb_name, obj_name, self.sum_gt[i],
                    len(self.tp[i]), np.sum(tp_all) / self.sum_gt[i],
                    0, 0, 0, 0, 0, 0, 0
                ])
            else:
                fps = []
                fp_cls_count += 1
                for i_fp, fp_item in enumerate([self.fp1, self.fp2, self.fp3, self.fp4, self.fp5, self.fp6, self.fp7]):
                    fp_ = np.array(fp_item[i], dtype=np.float32)[score_idx][:num_inst]
                    fp_num = int(np.sum(fp_))
                    fp_frac = fp_num / (num_inst - np.sum(tp))
                    fp_counts[i_fp] += fp_num
                    fp_frac_sum[i_fp] += fp_frac
                    fp_frac_matrix[i, i_fp] = fp_frac
                    fps.append(fp_frac)
                result_seq.append([i, vb_name, obj_name, self.sum_gt[i],
                    len(self.tp[i]), np.sum(tp_all) / self.sum_gt[i],
                    fps[0], fps[1], fps[2], fps[3], fps[4], fps[5], fps[6],
                ])
        # print result
        for res in sorted(result_seq, key=lambda k: k[5], reverse=True):  # sorted by recall score
            print(print_template.format(*res))
        print("{} of 600 classes with prediction were counted for error.".format(fp_cls_count))
        for i_fp in range(7):
            print("fp{} [{}] {:.04f}".format(i_fp + 1, fp_types[i_fp], fp_frac_sum[i_fp] / fp_cls_count))
        if demo_dir:
            if not os.path.exists(demo_dir):
                os.makedirs(demo_dir)
            # plot pie graph for error distribution
            # fp_labels = ["{}\n{}".format(fp_types[i_], fp_counts[i_]) for i_ in range(7)]
            fp_labels = ["{}".format(fp_counts[i_]) for i_ in range(7)]
            # neglect the second last error type for there is no instance of that type
            fp_counts = fp_counts[:-3] + fp_counts[-2:]
            fp_labels = fp_labels[:-3] + fp_labels[-2:]
            fp_sum = sum(fp_counts)
            plt.pie([fp_ / fp_sum for fp_ in fp_counts], labels=fp_labels, autopct='%2.1f%%', textprops={'fontsize': 20})
            plt.title("TP: {}, FP: {}".format(tp_sum, fp_sum), fontsize=24)
            graph_path = os.path.join(demo_dir, "diag_hico_thres0.01.png")
            # plt.legend(loc="lower left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.savefig(graph_path, bbox_inches='tight')
            # save result to csv file
            csv_path = os.path.join(demo_dir, "diag_hico_thres0.01.csv")
            with open(csv_path, "w", newline="") as f_:
                writer = csv.writer(f_)
                writer.writerow(["verb_id", "vb_name", "obj_name", "num_gt", "num_pred", "recall"] + fp_types)
                # writer.writerows(result_seq)
                for res in sorted(result_seq, key=lambda k: k[5], reverse=True):
                    writer.writerow(res[:5] + [round(item, 4) for item in res[5:]])
                total_gt = sum(self.sum_gt.values())
                total_pred = sum([len(tp) for tp in self.tp])
                tmp = [res[5] for res in result_seq]
                mean_recall = sum(tmp) / len(tmp)
                writer.writerow(["", "TOTAL", "TOTAL", total_gt, total_pred, round(mean_recall, 4)] + \
                    [round(frac / fp_cls_count, 4) for frac in fp_frac_sum])
            print("Error diagnosis result saved to {} and {}".format(csv_path, graph_path))

    def set_tp_fps_values(self, true_idx, v_id):
        for i, item in enumerate((self.tp, self.fp1, self.fp2, self.fp3, self.fp4, self.fp5, self.fp6, self.fp7)):
            if i == true_idx:
                item[v_id].append(1)
            else:
                item[v_id].append(0)

    def diagnose_img_error(self, pred_hoi, gt_hoi, match_pairs1, match_pairs2, pred_bbox, bbox_ov1, bbox_ov2):
        r"""Generate FP(different types)/TP lists for one image using `pred_hoi` and `gt_hoi`.

        Args:
            pred_hoi (dict): pred["hoi_prediction"]
            gt_hoi (dict): gt["hoi_annotation"]
            match_pairs1, match_pairs2 (dict): {pred_id: [list of matched gt id]}
            for ov threshold of 0.5 and 0.1 separately
            pred_bbox (list of dict): pred["predictions"]
            bbox_ov1, bbox_ov2 (dict): {pred_id: [list of IOU value with matched gt]}
            for ov threshold of 0.5 and 0.1 separately
        """
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                if pred_hoi_i['score'] < 0.01:
                    continue
                if isinstance(pred_hoi_i['category_id'], str):
                    pred_hoi_i['category_id'] = int(pred_hoi_i['category_id'].replace('\n', ''))

                try:
                    triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'], pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                except:
                    import ipdb; ipdb.set_trace()
                if triplet not in self.verb_name_dict:
                    continue
                verb_id = self.verb_name_dict.index(triplet)

                if pred_hoi_i["subject_id"] not in match_pairs2:
                    # person bck, subject ov with gt <= 0.1
                    self.set_tp_fps_values(1, verb_id)
                elif pred_hoi_i["subject_id"] not in match_pairs1:
                    # person misloc, subject ov with gt > 0.1 & <= 0.5
                    self.set_tp_fps_values(2, verb_id)
                elif pred_hoi_i["object_id"] not in match_pairs2:
                    # obj mis-grouping, obj ov with gt <= 0.1
                    self.set_tp_fps_values(3, verb_id)
                elif pred_hoi_i["object_id"] not in match_pairs1:
                    # obj misloc, obj ov with gt > 0.1 & <= 0.5
                    self.set_tp_fps_values(4, verb_id)
                else:
                    is_match = 0
                    # subject ov > 0.5, obj ov > 0.5
                    pred_sub_ids = match_pairs1[pred_hoi_i['subject_id']]
                    pred_obj_ids = match_pairs1[pred_hoi_i['object_id']]
                    pred_obj_ov = bbox_ov1[pred_hoi_i['object_id']]
                    pred_sub_ov = bbox_ov1[pred_hoi_i['subject_id']]
                    pred_category_id = pred_hoi_i['category_id']
                    max_ov = 0
                    max_gt_id = 0
                    # for each predicted hoi, if there are two gt matched,
                    # pick the gt with the largest object score
                    for gt_id in range(len(gt_hoi)):
                    # for gt_id in np.where(vis_tag==0)[0]:
                        gt_hoi_i = gt_hoi[gt_id]
                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            min_ov_gt = min(pred_sub_ov[pred_sub_ids.index(gt_hoi_i['subject_id'])],
                                pred_obj_ov[pred_obj_ids.index(gt_hoi_i['object_id'])])
                            if min_ov_gt > max_ov:
                                max_ov = min_ov_gt
                                max_gt_id = gt_id
                    if is_match == 1:
                        if vis_tag[max_gt_id] == 0:
                            # TP
                            self.set_tp_fps_values(0, verb_id)
                            vis_tag[max_gt_id] = 1
                        else:
                            # duplicate
                            self.set_tp_fps_values(7, verb_id)
                    else:
                        # p ov > 0.5, o ov > 0.5, incorrect action
                        self.set_tp_fps_values(6, verb_id)
                self.diag_score[verb_id].append(pred_hoi_i["score"])
        # else:
        #     import ipdb; ipdb.set_trace()


def save_aps_to_csv(demo_dir, verb_name_dict, ap, max_recall, category_info, num_gt, r_inds, c_inds, ap_2=None, max_recall_2=None):
    r"""Save AP and max_recall to csv files
    """
    hoi_c, obj_c, vb_c = category_info
    result_seq = []
    item_names = ["verb_id", "vb_name", "obj_name", "num_gt", "ap", "max_recall"]
    if ap_2 is not None and max_recall_2 is not None:
        item_names.extend(["ap_2", "max_recall_2", "ap_diff", "max_recall_diff"])
    for i in range(len(verb_name_dict)):
        res = [
            i,
            vb_c[verb_name_dict[i][2]-1],
            obj_c[verb_name_dict[i][1]-1],
            num_gt[i],
            round(ap[i], 4),
            round(max_recall[i], 4)
        ]
        if ap_2 is not None and max_recall_2 is not None:
            res.extend([
                round(ap_2[i], 4),
                round(max_recall_2[i], 4),
                round(ap_2[i] - ap[i], 4),
                round(max_recall_2[i] - max_recall[i], 4)
            ])
        result_seq.append(res)

    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    # save ap (all) to csv
    csv_path = os.path.join(demo_dir, "eval_hico.csv")
    with open(csv_path, "w", newline="") as f_:
        writer = csv.writer(f_)
        writer.writerow(item_names)
        writer.writerows(result_seq)
    print("Eval result saved to {}".format(csv_path))
    # save ap (rare classes) to csv
    csv_path = os.path.join(demo_dir, "eval_hico_rare.csv")
    with open(csv_path, "w", newline="") as f_:
        writer = csv.writer(f_)
        writer.writerow(item_names)
        for i in range(len(verb_name_dict)):
            if i in r_inds:
                writer.writerow(result_seq[i])
    print("Rare eval result saved to {}".format(csv_path))
    # save ap (non-rare classes) to csv
    csv_path = os.path.join(demo_dir, "eval_hico_nonrare.csv")
    with open(csv_path, "w", newline="") as f_:
        writer = csv.writer(f_)
        writer.writerow(item_names)
        for i in range(len(verb_name_dict)):
            if i in c_inds:
                writer.writerow(result_seq[i])
    print("Non-rare eval result saved to {}".format(csv_path))

def plot_pr_curve_diff(pre1, rec1, pre2, rec2, score1, score2, fnames1, fnames2, verb_name_dict, demo_dir=None):
    for i in tqdm(range(len(verb_name_dict))):
        plt.plot(rec1[i][1:], pre1[i][1:], label='QPIC Precision-Recall', color="red")
        plt.plot(rec2[i][1:], pre2[i][1:], label='QPIC+ours Precision-Recall', color="blue")
        # if i == 10:
        #     import ipdb; ipdb.set_trace()
        s1 = np.concatenate(([0.], np.sort(score1[i])[::-1], [0.]))
        s2 = np.concatenate(([0.], np.sort(score2[i])[::-1], [0.]))
        plt.plot(rec1[i][1:], s1[1:], label='QPIC score-Recall', color="red", marker='o', markersize=2, linestyle="dashed")
        plt.plot(rec2[i][1:], s2[1:], label='QPIC+ours score-Recall', color='blue', marker='o', markersize=2, linestyle="dashed")
        # fn1 = [""] + [fnames1[i][idx] for idx in np.argsort(score1[i])[::-1].tolist()] + [""]
        # fn2 = [""] + [fnames2[i][idx] for idx in np.argsort(score2[i])[::-1].tolist()] + [""]
        # csv_path = os.path.join(demo_dir, "{}.csv".format(i))
        # with open(csv_path, "w", newline="") as f_:
        #     writer = csv.writer(f_)
        #     writer.writerow(["pre1", "rec1", "pre2", "rec2", "s1", "s2", "fn1", "fn2"])
        #     writer.writerows([[p1, r1, p2, r2, s_1, s_2, f1, f2] for p1, r1, p2, r2, s_1, s_2, f1, f2 in zip(pre1[i], rec1[i], pre2[i], rec2[i], s1, s2, fn1, fn2)])
        # for r, s, f in zip(rec1[i], s1, fn1):
        #     plt.text(r, s, "{} - {:.2f}".format(f, s), fontsize='x-small')
        # for r, s, f in zip(rec2[i], s2, fn2):
        #     plt.text(r, s, "{} - {:.2f}".format(f, s), fontsize='x-small')
        plt.legend(fontsize=16)
        plt.savefig(os.path.join(demo_dir, "{}.pdf".format(i)), bbox_inches='tight')
        plt.clf()



def plot_average_score_recall_curve(pres, recs, score, label=None):
    avg_s_list = []
    for t in np.arange(0., 1.1, 0.1):
        s_at_r_list = []
        for i in range(600):
            rec = recs[i][1:-1]
            pre = pres[i][1:-1]
            sorted_score = np.sort(score[i])[::-1]
            if np.sum(rec >= t) == 0:
                s_at_r = 0
            else:
                s_at_r = np.max(sorted_score[rec >= t])
            s_at_r_list.append(s_at_r)
        avg_s = sum(s_at_r_list) / len(s_at_r_list)
        avg_s_list.append(avg_s)
    # plt.plot(np.arange(0, 1.1, 0.1), avg_s_list, label=label, marker="o")
    return avg_s_list


def plot_average_pre_recall_curve(pres, recs, label=None):
    avg_s_list = []
    for t in np.arange(0., 1.1, 0.1):
        s_at_r_list = []
        for i in range(600):
            rec = recs[i][1:-1]
            pre = pres[i][1:-1]
            if np.sum(rec >= t) == 0:
                s_at_r = 0
            else:
                s_at_r = np.max(pre[rec >= t])
            s_at_r_list.append(s_at_r)
        avg_s = sum(s_at_r_list) / len(s_at_r_list)
        avg_s_list.append(avg_s)
    # plt.plot(np.arange(0, 1.1, 0.1), avg_s_list, label=label, linestyle="dashed")
    return avg_s_list


def plot_average_score_recall_curve_selected(pres, recs, score, idx, label=None):
    avg_s_list = []
    for t in np.arange(0., 1.1, 0.1):
        s_at_r_list = []
        for i in range(600):
            if i not in idx:
                continue
            rec = recs[i][1:-1]
            pre = pres[i][1:-1]
            sorted_score = np.sort(score[i])[::-1]
            if np.sum(rec >= t) == 0:
                s_at_r = 0
            else:
                s_at_r = np.max(sorted_score[rec >= t])
            s_at_r_list.append(s_at_r)
        avg_s = sum(s_at_r_list) / len(s_at_r_list)
        avg_s_list.append(avg_s)
    # plt.plot(np.arange(0, 1.1, 0.1), avg_s_list, label=label, marker="o")
    return avg_s_list


def plot_average_pre_recall_curve_selected(pres, recs, idx, label=None):
    avg_s_list = []
    for t in np.arange(0., 1.1, 0.1):
        s_at_r_list = []
        for i in range(600):
            if i not in idx:
                continue
            rec = recs[i][1:-1]
            pre = pres[i][1:-1]
            if np.sum(rec >= t) == 0:
                s_at_r = 0
            else:
                s_at_r = np.max(pre[rec >= t])
            s_at_r_list.append(s_at_r)
        avg_s = sum(s_at_r_list) / len(s_at_r_list)
        avg_s_list.append(avg_s)
    # plt.plot(np.arange(0, 1.1, 0.1), avg_s_list, label=label, linestyle="dashed")
    return avg_s_list



def plot_allcat_pr_sr_curve(tpfp1, tpfp2, score1, score2, sum_gt):
    tp1, fp1 = tpfp1
    tp2, fp2 = tpfp2
    total_gt = sum(sum_gt.values())
    prec1, rec1, s1_sorted, tp1_sorted, fp1_sorted = calc_prec_rec_allcat(tp1, fp1, score1, total_gt)
    prec2, rec2, s2_sorted, tp2_sorted, fp2_sorted = calc_prec_rec_allcat(tp2, fp2, score2, total_gt)
    r1_points, p1_points, s1_points = get_pr_10points(rec1, prec1, s1_sorted)
    r2_points, p2_points, s2_points = get_pr_10points(rec2, prec2, s2_sorted)

    # plt.plot(r1_points, s1_points, marker='o', label="QPIC SR")
    # plt.plot(r1_points, p1_points, linestyle="dashed", label="QPIC PR")
    # plt.plot(r2_points, s2_points, marker='o', label="Ours SR")
    # plt.plot(r2_points, p2_points, linestyle="dashed", label="Ours PR")

    plt.plot(r2_points, s2_points - s1_points, marker='o', markersize=8, label="SR-DIFF")
    plt.plot(r2_points, p2_points - p1_points, linestyle="dashed", label="PR-DIFF")


def get_pr_10points(r, p, s):
    r_points = np.arange(0.0, 1.03, 0.03)
    p_list = []
    s_list = []
    for t in r_points:
        if np.sum(r >= t) == 0:
            p_at_r = 0
            s_at_r = 0
        else:
            # p_at_r = np.max(p[r >= t])
            # s_at_r = np.max(s[r >= t])
            p_at_r_loc = np.argmax(p[r >= t])
            # s_at_r_loc = np.argmax(s[r >= t])
            p_at_r = p[r >= t][p_at_r_loc]
            s_at_r = s[r >= t][p_at_r_loc]
        p_list.append(p_at_r)
        s_list.append(s_at_r)
    return r_points, np.array(p_list), np.array(s_list)


def calc_prec_rec_allcat(tp, fp, score, n_gt):
    tp_list, fp_list, s_list = [], [], []
    for i in sorted(tp.keys()):
        tp_list.extend(tp[i])
        fp_list.extend(fp[i])
        s_list.extend(score[i])
    tp_list = np.array(tp_list)
    fp_list = np.array(fp_list)
    s_list = np.array(s_list)
    # import ipdb; ipdb.set_trace()
    s_idx = np.argsort(s_list)[::-1]
    tp_tmp = tp_list[s_idx].cumsum()
    fp_tmp = fp_list[s_idx].cumsum()
    prec1 = tp_tmp / (tp_tmp + fp_tmp)
    rec1 = tp_tmp / n_gt
    return prec1, rec1, s_list[s_idx], tp_list[s_idx], fp_list[s_idx]


if __name__ == "__main__":
    print("Starting HOI detection result mAP evaluation on HICO-DET dataset.")
    hoi_eval = HICO(args.gt_path, args.hico_list_dir)
    category_info = load_hico_category(args.hico_list_dir)
    hoi_c, obj_c, vb_c = category_info
    # get gt information
    verb_name_dict = hoi_eval.get_verb_name_dict()
    sum_gt = hoi_eval.get_sum_gt()
    num_gt = hoi_eval.get_sum_gt()
    r_inds, c_inds = hoi_eval.get_rare_nonrare_inds()

    model1 = args.det_path
    output_hoi = load_res_from_json(model1)
    model_name = model1.split("/")[-2]
    demo_dir = os.path.join(args.root_dir, model_name)
    # perform `diag_hico`
    if args.diag:
        hoi_eval.diagnosis(output_hoi, demo_dir=demo_dir)
        hoi_eval.reset()
    else:
        # perform `eval_hico`
        mAP = hoi_eval.evalution(output_hoi, verbose=not args.diff_ap)
        print("Calculated mAP for model {} is {}".format(model1, mAP))
        ap = hoi_eval.get_ap()
        max_recall = hoi_eval.get_max_recall()
        pres = hoi_eval.pres
        recs = hoi_eval.recs
        score = hoi_eval.score
        fnames = hoi_eval.fnames
        tp = hoi_eval.tp
        fp = hoi_eval.fp
        hoi_eval.reset()

        # plot mA-score curve, 11 points, each points averaged over all classes
        # plot_average_score_recall_curve(pres, recs, score, label="baseline SR")
        # plot_average_pre_recall_curve(pres, recs, label="baseline PR")
        # plt.savefig(model_name + ".jpg")
        # print(model_name)

        # import ipdb; ipdb.set_trace()
        # get TP scores
        # s_i_list = []
        # s_i_avg_list = []
        # for i in range(600):
        #     s_i = np.array(score[i])
        #     t_i = np.array(tp[i])
        #     t_idx = t_i.astype('bool')
        #     s_i_tp = s_i[t_idx]
        #     s_i_list.append(s_i_tp)
        #     if len(s_i_tp) > 0:
        #         s_i_avg_list.append(s_i_tp.mean())
        # print(s_i_list[500].shape)
        # all_s_tp = np.concatenate(s_i_list, axis=0)
        # all_s_tp = np.sort(all_s_tp)[::-1]
        # print("1000th, 2000th, 5000th, 10000th, 15000th, 20000th", all_s_tp[1000], all_s_tp[2000], all_s_tp[5000], all_s_tp[10000], all_s_tp[15000], all_s_tp[20000])
        # import ipdb; ipdb.set_trace()
        # print(all_s_tp.shape)
        # print(all_s_tp.mean())
        # print(len(s_i_avg_list))
        # print(sum(s_i_avg_list) / len(s_i_avg_list))
        # if not args.diff_ap:
        #     # save ap and max_recall to csv
        #     save_aps_to_csv(demo_dir, verb_name_dict, ap, max_recall, category_info, num_gt, r_inds, c_inds)
        #     hoi_eval.reset()
        #     known_mAP = hoi_eval.evalution_known_object(output_hoi, verbose=False)
        # hoi_eval.reset()

#         # perform `diff_ap`
        if args.diff_ap:
            model2 = args.det2_path
            assert model2, "for diff ap, argument det_result_2 required"

            output_hoi_2 = load_res_from_json(model2)
            mAP_2 = hoi_eval.evalution(output_hoi_2, verbose=False)
            print("Calculated mAP for model {} is {}".format(model2, mAP_2))
            ap_2 = hoi_eval.get_ap()
            max_recall_2 = hoi_eval.get_max_recall()
            pres2 = hoi_eval.pres
            recs2 = hoi_eval.recs
            score2 = hoi_eval.score
            fnames2 = hoi_eval.fnames
            tp2 = hoi_eval.tp
            fp2 = hoi_eval.fp
            hoi_eval.reset()

            # import ipdb; ipdb.set_trace()
            # m1_msr = plot_average_score_recall_curve(pres, recs, score, label="QPIC MeanScore-Recall")
            # m2_msr = plot_average_score_recall_curve(pres2, recs2, score2, label="QPIC+ours MeanScore-Recall")
            # m1_mpr = plot_average_pre_recall_curve(pres, recs, label="QPIC MeanPrecision-Recall")
            # m2_mpr = plot_average_pre_recall_curve(pres2, recs2, label="QPIC+ours MeanPrecision-Recall")
            # plt.plot(np.arange(0.0, 1.1, 0.1), np.array(m2_msr) - np.array(m1_msr), label="MeanScoreDiff-Recall", marker="o")
            # plt.plot(np.arange(0.0, 1.1, 0.1), np.array(m2_mpr) - np.array(m1_mpr), label="MeanPrecisonDiff-Recall", linestyle="dashed")
            plt.axhline(y=0, color='black', linestyle="-")
            pos_diff_categories = np.where((ap_2 - ap) >= 0)[0]
            m1_msr = plot_average_score_recall_curve_selected(pres, recs, score, pos_diff_categories, label="QPIC MS-R")
            m2_msr = plot_average_score_recall_curve_selected(pres2, recs2, score2, pos_diff_categories, label="QPIC+ours MS-R")
            m1_mpr = plot_average_pre_recall_curve_selected(pres, recs, pos_diff_categories, label="QPIC MP-R")
            m2_mpr = plot_average_pre_recall_curve_selected(pres2, recs2, pos_diff_categories, label="QPIC+ours MP-R")
            plt.plot(np.arange(0.0, 1.1, 0.1), np.array(m2_msr) - np.array(m1_msr), label="MSD-R", marker="o")
            plt.plot(np.arange(0.0, 1.1, 0.1), np.array(m2_mpr) - np.array(m1_mpr), label="MPD-R", linestyle="dashed")
            plt.legend(fontsize=16)
            plt.savefig(model_name + "_diff_rise_showdiff.pdf", bbox_inches='tight')
            # print(model_name)

            # plot_allcat_pr_sr_curve([tp, fp], [tp2, fp2], score, score2, sum_gt)
            # plt.legend(fontsize=12)
            # plt.savefig("search_MPDR_models/" + model_name + model2.split('/')[-2] + model2.split('.')[-1] + "_diff_plotdiff_10points.pdf", bbox_inches='tight')

#             # print model AP & max_recall diff results
#             print("Diff max_recall for model {} and {}:".format(model1, model2))
#             for i in range(len(verb_name_dict)):
#                 print('class {} {:5.5}-{:10.10} | -- max_rec_1: {:.04f} -- | -- max_rec_2: {:.04f} --| -- max_rec_diff: {:.04f} -- |'.format(
#                     i, vb_c[verb_name_dict[i][2]-1], obj_c[verb_name_dict[i][1]-1], max_recall[i], max_recall_2[i], max_recall_2[i] - max_recall[i]
#                 ))
#             print("Diff AP for model {} and {}:".format(model1, model2))
#             for i in range(len(verb_name_dict)):
#                 print('class {} {:5.5}-{:10.10} | -- ap_1: {:.04f} -- | -- ap_2: {:.04f} --| -- ap_diff: {:.04f} -- |'.format(
#                     i, vb_c[verb_name_dict[i][2]-1], obj_c[verb_name_dict[i][1]-1], ap[i], ap_2[i], ap_2[i] - ap[i]
#                 ))
#             print('class --- {:16.16} | -- ap_1: {:.04f} -- | -- ap_2: {:.04f} --| -- ap_diff: {:.04f} -- |'.format(
#                 "TOTAL", mAP, mAP_2, mAP_2 - mAP
#             ))
#             plot_pr_curve_diff(pres, recs, pres2, recs2, score, score2, fnames, fnames2, verb_name_dict, demo_dir=demo_dir)
#             save_aps_to_csv(demo_dir, verb_name_dict, ap, max_recall, category_info, num_gt, r_inds, c_inds, ap_2, max_recall_2)

# # TODO: miss_gt type


