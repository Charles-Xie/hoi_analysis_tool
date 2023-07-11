# to run this file
# see Makefile
# python3 src/stat_hico.py --config config/stat.yaml

import os
from collections import Counter, OrderedDict, defaultdict

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from utils import VALID_OBJ_IDS
from utils import load_res_from_json, load_hico_category
from argparser import args
from eval_detection import get_AP_dict
from eval_hico import HICO

def count_obj_from_gt(gt_result):
    obj_ids = [det_item["category_id"] for im_result in gt_result for det_item in im_result["annotations"]]
    counted = Counter(obj_ids)
    for obj_id in VALID_OBJ_IDS:
        if obj_id not in counted:
            counted[obj_id] = 0
    return counted

def count_vb_from_gt(gt_result):
    vb_ids = [hoi_item["category_id"] for im_result in gt_result for hoi_item in im_result["hoi_annotation"]]
    counted = Counter(vb_ids)
    for vb_id in range(1, 118):
        if vb_id not in counted:
            counted[vb_id] = 0
    return counted

def plot_bar_from_dict(count_dict, x_label_list, save_name):
    plt.bar(range(len(count_dict)), count_dict.values())
    x_labels = [x_label_list[i - 1] for i in count_dict.keys()]
    plt.xticks(range(len(count_dict)), x_labels, rotation=70, fontsize=2)
    for idx, val in enumerate(count_dict.values()):
        plt.text(idx, val, str(val), fontdict={"fontsize": 1})
    # check path and save image file
    save_dir = os.path.dirname(save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_name, dpi=1200)
    plt.close()

def plot_bars_from_multi_dict(dicts, x_label_list, save_name):
    for count_dict in dicts:
        plt.bar(range(len(count_dict)), count_dict.values())
        for idx, val in enumerate(count_dict.values()):
            plt.text(idx, val, str(val), fontdict={"fontsize": 1})
    x_labels = [x_label_list[i - 1] for i in count_dict.keys()]
    plt.xticks(range(len(count_dict)), x_labels, rotation=70, fontsize=2)
    # check path and save image file
    save_dir = os.path.dirname(save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_name, dpi=1200)
    plt.close()


if __name__ == "__main__":
    # should use merged (filtered) det annotations
    # gt_result = load_res_from_json(args.gt_path)  # we use training gt, not testing gt
    train_gt_path = os.path.join(args.root_dir, "annotations", "trainval_hico.json")
    gt_result  = load_res_from_json(train_gt_path)
    hoi_c, obj_c, vb_c = load_hico_category(args.hico_list_dir)
    save_dir = os.path.join(args.root_dir, "stat_hico")

    # region GTObjInfo
    gt_obj_count = count_obj_from_gt(gt_result)
    gt_obj_count_ordered = OrderedDict(sorted(gt_obj_count.items(), key=lambda kv: kv[1], reverse=True))
    # plot object distribution
    plot_bar_from_dict(gt_obj_count_ordered, obj_c, os.path.join(save_dir, "gt_obj_count.png"))
    # endregion

    # region GTVerbInfo
    gt_vb_count = count_vb_from_gt(gt_result)
    gt_vb_count_ordered = OrderedDict(sorted(gt_vb_count.items(), key=lambda kv: kv[1], reverse=True))
    # plot verb distribution
    plot_bar_from_dict(gt_vb_count_ordered, vb_c, os.path.join(save_dir, "gt_vb_count.png"))
    # endregion

    # region GTHOIInfo
    hoi_eval = HICO(args.gt_path, args.hico_list_dir)
    verb_name_dict = hoi_eval.get_verb_name_dict()
    # from test set
    # num_gt = hoi_eval.get_sum_gt()
    # from train set
    num_gt = hoi_eval.get_train_sum_gt()
    tmp = OrderedDict(sorted(num_gt.items(), key=lambda x: x[1], reverse=True))
    gt_hoi_count_ordered = OrderedDict({k+1: v for k, v in tmp.items()})
    # plot HOI distribution
    plot_bar_from_dict(gt_hoi_count_ordered, hoi_c, os.path.join(save_dir, "gt_hoi_count.png"))
    # endregion

    # region DetectorFinetuningComparsion
    # eval det result on head and tail object classes
    ap = get_AP_dict(args.det_path, args.gt_path, args.hico_list_dir)
    # sort using obj count
    ap_ordered = OrderedDict([(k, ap[k]) for k in gt_obj_count_ordered.keys()])
    # plot the mAP on different classes
    plot_bar_from_dict(ap_ordered, obj_c, os.path.join(save_dir, "det_ap.png"))
    if args.det2_path:
        ap2 = get_AP_dict(args.det2_path, args.gt_path, args.hico_list_dir)
        ap2_ordered = OrderedDict([(k, ap2[k]) for k in gt_obj_count_ordered.keys()])
        plot_bars_from_multi_dict([ap2_ordered, ap_ordered], obj_c, os.path.join(save_dir, "det_ap_diff.png"))
    # endregion

    # region HOIDetectionHeadTailComparison
    pred_result = load_res_from_json(args.det_path)
    hoi_eval.evalution(pred_result, verbose=False)
    hoi_ap = hoi_eval.get_ap()
    # plot by 117 verb
    vb_map = defaultdict(list)
    for i, triplet in enumerate(verb_name_dict):
        vb_map[triplet[2]].append(hoi_ap[i])
    vb_map = dict(vb_map)
    vb_map_ordered = OrderedDict([(k, vb_map[k]) for k in gt_vb_count_ordered.keys()])
    vb_map_avg_ordered = OrderedDict([(k, sum(vb_map[k]) / len(vb_map[k])) for k in gt_vb_count_ordered.keys()])
    plot_bar_from_dict(vb_map_avg_ordered, vb_c, os.path.join(save_dir, "verb_ap.png"))
    # plot by 600 hoi
    hoi_map_ordered = OrderedDict([(k, hoi_ap[k-1]) for k in gt_hoi_count_ordered.keys()])
    plot_bar_from_dict(hoi_map_ordered, hoi_c, os.path.join(save_dir, "hoi_ap.png"))
    hoi_eval.reset()
    # endregion

    #region HOIDetectionModelCompairson
    if args.det2_path:
        pred_result2 = load_res_from_json(args.det2_path)
        hoi_eval.evalution(pred_result2, verbose=False)
        hoi_ap2 = hoi_eval.get_ap()
        # plot by 117 verb
        vb_map2 = defaultdict(list)
        for i, triplet in enumerate(verb_name_dict):
            vb_map2[triplet[2]].append(hoi_ap2[i])
        vb_map2 = dict(vb_map2)
        vb_map_ordered2 = OrderedDict([(k, vb_map2[k]) for k in gt_vb_count_ordered.keys()])
        vb_map_avg_ordered2 = OrderedDict([(k, sum(vb_map2[k]) / len(vb_map2[k])) for k in gt_vb_count_ordered.keys()])
        plot_bars_from_multi_dict([vb_map_avg_ordered2, vb_map_avg_ordered], vb_c, os.path.join(save_dir, "verb_ap_diff.png"))
