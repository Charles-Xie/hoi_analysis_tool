from collections import defaultdict

from utils import load_json

# GT_PATH = "./hico/annotations/test_hico.json"
GT_PATH = "./hico/annotations/trainval_hico.json"

def get_id(img_name):
    return int(img_name.split(".")[-2].split("_")[-1])

img_h_multi_obj_cats = set()
img_h_multi_objs = set()

gt_path = GT_PATH
gt_result = load_json(gt_path)
for gt in gt_result:
    ho_num_counter = defaultdict(set)
    ho_cat_counter = defaultdict(set)
    objs = gt["annotations"]
    # one human, multiple object
    for hoi in gt["hoi_annotation"]:
        ho_num_counter[hoi["subject_id"]].add(hoi["object_id"])
        ho_cat_counter[hoi["subject_id"]].add(objs[hoi["object_id"]]["category_id"])
    for h_id, o_ids in dict(ho_num_counter).items():
        if len(o_ids) >= 3:
            print(h_id, o_ids, gt["file_name"])
            img_h_multi_objs.add(get_id(gt["file_name"]))
    for h_id, o_cats in dict(ho_cat_counter).items():
        if len(o_cats) >= 2:
            # print(h_id, o_cats, gt["file_name"]) 
            img_h_multi_obj_cats.add(get_id(gt["file_name"]))

print("Find {} images: {}".format(len(list(img_h_multi_objs)), list(img_h_multi_objs)))
print("First 100: ".format(list(img_h_multi_objs)[:100]))

# print("Find {} images: {}".format(len(list(img_h_multi_obj_cats)), list(img_h_multi_obj_cats)))
