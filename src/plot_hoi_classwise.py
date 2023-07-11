import os

from tqdm import tqdm

from argparser import args
from utils import VALID_OBJ_IDS
from eval_hico import HICO
from utils import load_res_from_json, load_hico_category, create_classwise_generator_by_det, create_classwise_generator_by_gt, multi_process_image, reorganize_func_1st, reorganize_func_2nd
from plot_hoi import add_match_key, vis_hoi_tpfp_by_category

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
    hoi_eval = HICO(args.gt_path, args.hico_list_dir)
    verb_name_dict = hoi_eval.get_verb_name_dict()
    num_gt_train = hoi_eval.get_train_sum_gt()
    # feature `vis_hoi_class`:
    # visualize HOI (FP, TP, False, Missed) by vb and obj class.
    vb_name_selected, obj_name_selected = None, None
    if args.class_name:
        assert "~" in args.class_name, (
            "'~' is required in class_name as separator between verb_name and object_name. You can use 'adjust~', '~tie' or 'adjust~tie'")
        vb_name_selected, obj_name_selected = args.class_name.split("~")
    # feature `vis_hoi_class`:
    # for False and FP pairs, use [lower, upper] score range for verb predictions.
    lower, upper = args.score_range
    if vb_name_selected:
        print("Verb name {} selected for visualization.".format(vb_name_selected))
    else:
        print("Verb name not provided. Visualize all verb classes.")
    if obj_name_selected:
        print("Object name {} selected for visualization.".format(obj_name_selected))
    else:
        print("Object name not provided. Visualize all object classes.")
    for vb_id in range(1, len(vb_c) + 1):
        vb_name = vb_c[vb_id - 1]
        if vb_name_selected and vb_name != vb_name_selected:
            continue
        for obj_id in VALID_OBJ_IDS:
            obj_name = obj_c[obj_id - 1]
            if obj_name_selected and obj_name != obj_name_selected:
                continue
            category_dir = os.path.join(demo_dir, "hoi_analysis_classwise", vb_name, obj_name)
            if not vb_name_selected and obj_name_selected:
                # if only obj_name are provided, plot in obj_vb dirs
                category_dir = os.path.join(demo_dir, "hoi_analysis_classwise", obj_name, vb_name)
            try:
                hoi_id = verb_name_dict.index([1, obj_id, vb_id])
            except ValueError:
                continue
            print("HOI result visualization for {} - {} ({} - {}) starting. See {}".format(vb_name, obj_name, vb_id, obj_id, category_dir))
            if num_gt_train[hoi_id] < 10:
                # rare split
                create_generator = create_classwise_generator_by_det
            else:
                # non-rare split
                create_generator = create_classwise_generator_by_gt
            gt_det_classwise_gen = create_generator(gt_result, updated_pred, vb_id, obj_id)
            gt_det_list = list(gt_det_classwise_gen)
            if multi_process and len(gt_det_list) > 10:
                gt_list, det_list = zip(*gt_det_list)
                multi_process_image(
                    vis_hoi_tpfp_by_category,
                    det_list, reorganize_func_2nd, 16,
                    gt_list, vb_id, obj_id, category_info,
                    root_dir, category_dir, lower, upper,
                )
            else:
                for gt, det in tqdm(gt_det_list):
                    if len(det['hoi_prediction']) == 0:
                        continue
                    vis_hoi_tpfp_by_category(
                        det, gt, vb_id, obj_id, category_info,
                        root_dir, category_dir, lower, upper,
                    )
            print("HOI result visualization for {} - {} ({} - {}) done. See {}".format(vb_name, obj_name, vb_id, obj_id, category_dir))
