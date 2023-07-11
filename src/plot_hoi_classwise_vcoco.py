import os

from tqdm import tqdm

from argparser import args
from utils import VALID_OBJ_IDS, VALID_ACTION_IDS_VCOCO
from utils import load_res_from_json, load_hico_category, create_classwise_generator_by_det, create_classwise_generator_by_gt, multi_process_image, reorganize_func_2nd
from plot_hoi import add_match_key, vis_hoi_tpfp_by_category

def traverse_gt_det_list(gt_det_generator):
    gt_det_list = []
    tp_interaction_score_list = []
    for gt, det in gt_det_generator:
        gt_det_list.append((gt, det))
        for hoi in det["hoi_prediction"]:
            if hoi["match"] is not None and hoi["match"] >= 0:  # TP
                sub = det["predictions"][hoi["subject_id"]]
                obj = det["predictions"][hoi["object_id"]]
                interaction_score = hoi["score"] / (sub["score"] * obj["score"])
                tp_interaction_score_list.append(interaction_score)
    return gt_det_list, min(tp_interaction_score_list)
        

if __name__ == "__main__":
    multi_process = True
    print("Starting HOI result plotting.")
    root_dir = args.root_dir
    model_name = args.det_path.split("/")[-2] + "min_tp_score"
    demo_dir = os.path.join(root_dir, model_name)
    gt_result = load_res_from_json(args.gt_path)
    pred_result = load_res_from_json(args.det_path)
    updated_pred = add_match_key(gt_result, pred_result)
    category_info = load_hico_category(args.hico_list_dir)
    hoi_c, obj_c, vb_c = category_info
    # feature `vis_hoi_class`:
    # visualize HOI (FP, TP, False, Missed) by vb class.
    vb_name_selected = None
    if args.class_name:
        vb_name_selected = args.class_name
    # feature `vis_hoi_class`:
    # for False and FP pairs, use [lower, upper] score range for verb predictions.
    lower, upper = args.score_range
    if vb_name_selected:
        print("Verb name {} selected for visualization.".format(vb_name_selected))
    else:
        print("Verb name not provided. Visualize all verb classes.")
    obj_id = None
    for vb_id in VALID_ACTION_IDS_VCOCO:
        vb_name = vb_c[vb_id - 1]
        if vb_name_selected and vb_name != vb_name_selected:
            continue
        category_dir = os.path.join(demo_dir, "hoi_analysis_classwise", vb_name)
        print("HOI result visualization for {} (action {}) starting. See {}".format(vb_name, vb_id, category_dir))
        create_generator = create_classwise_generator_by_gt
        gt_det_classwise_gen = create_generator(gt_result, updated_pred, vb_id)
        # gt_det_list = list(gt_det_classwise_gen)
        gt_det_list, min_tp_score = traverse_gt_det_list(gt_det_classwise_gen)
        # if multi_process and len(gt_det_list) > 10:
        #     gt_list, det_list = zip(*gt_det_list)
        #     multi_process_image(
        #         vis_hoi_tpfp_by_category,
        #         det_list, reorganize_func_2nd, 16,
        #         gt_list, vb_id, obj_id, category_info,
        #         root_dir, category_dir, lower, upper,
        #     )
        # else:
        # plot for one class
        lower = min_tp_score
        for gt, det in tqdm(gt_det_list):
            if len(det['hoi_prediction']) == 0:
                print("[Warning] No hoi_prediction found for image {}".format(det["file_name"]))
                continue
            vis_hoi_tpfp_by_category(
                det, gt, vb_id, obj_id, category_info,
                root_dir, category_dir, lower, upper,
            )
        print("HOI result visualization for {} (action {}) done. See {}".format(vb_name, vb_id, category_dir))
