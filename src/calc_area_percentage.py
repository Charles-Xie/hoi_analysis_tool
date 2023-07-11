import os
import numpy as np
# import cv2
from PIL import Image
from tqdm import tqdm

from utils import load_json

def calc_ho_sum_area_percentage(h_box_list, o_box_list, img_size):
    r"""
    percentage = (human_area + object_area) / image_area
    """
    h_boxes = np.array(h_box_list).reshape(-1, 2, 2)
    o_boxes = np.array(o_box_list).reshape(-1, 2, 2)
    h_area = (h_boxes[:, 1, :] - h_boxes[:, 0, :]).prod(axis=1)
    o_area = (o_boxes[:, 1, :] - o_boxes[:, 0, :]).prod(axis=1)
    img_area = np.array(list(img_size)).prod()
    area_percent = (h_area + o_area) / img_area
    return area_percent

def calc_ho_nonrepeat_area_percentage(h_box_list, o_box_list, img_size):
    r"""
    percentage = human_object_area / image_area
    or:
    percentage = (human_area + object_area - overlapped_area) / image_area
    """
    h_boxes = np.array(h_box_list).reshape(-1, 2, 2)
    o_boxes = np.array(o_box_list).reshape(-1, 2, 2)
    h_area = (h_boxes[:, 1, :] - h_boxes[:, 0, :]).prod(axis=1)
    o_area = (o_boxes[:, 1, :] - o_boxes[:, 0, :]).prod(axis=1)
    # calculate overlapped area
    x0y0 = np.maximum(h_boxes[:, 0, :], o_boxes[:, 0, :])
    x1y1 = np.minimum(h_boxes[:, 1, :], o_boxes[:, 1, :])
    overlapped_wh = x1y1 - x0y0
    overlapped_wh = np.maximum(overlapped_wh, 0)
    overlapped_area = overlapped_wh.prod(axis=1)
    img_area = np.array(list(img_size)).prod()
    area_percent = (h_area + o_area - overlapped_area) / img_area
    return area_percent


GT_PATH = "./hico/annotations/test_hico.json"
IMG_DIR = "./hico/images/test2015"

gt_path = GT_PATH
img_dir = IMG_DIR
gt_result = load_json(gt_path)
area_percent_list = []
for gt in tqdm(gt_result, "processing image..."):
    # calc H-O pair area vs. image area
    img_path = os.path.join(img_dir, gt["file_name"])
    # img = cv2.imread(img_path)
    # if img is None:
    #     print("Image not found for {}".format(img_path))
    # height, width = img.shape[0], img.shape[1]
    img = Image.open(img_path)
    width, height = img.size
    gt_objs = gt["annotations"]
    h_box_list = [gt_objs[hoi["subject_id"]]["bbox"] for hoi in gt["hoi_annotation"]]
    o_box_list = [gt_objs[hoi["object_id"]]["bbox"] for hoi in gt["hoi_annotation"]]
    # ho_area_percent = calc_ho_sum_area_percentage(h_box_list, o_box_list, (width, height))
    ho_area_percent = calc_ho_nonrepeat_area_percentage(h_box_list, o_box_list, (width, height))
    area_percent_list.append(ho_area_percent)
print("Collected results from {} images".format(len(area_percent_list)))
ho_area_percents = np.concatenate(area_percent_list, axis=0)
print("Colllected results from {} h-o pairs".format(len(ho_area_percents)))
mean_ho_area_percent = ho_area_percents.mean()
import ipdb; ipdb.set_trace()
print("The average percentage of ho pair area vs. image area is {}".format(mean_ho_area_percent))
