# HOI Analysis Tool

Error analysis and result visualization tool for Human Object Interaction Detection task.

## Functionality

This tool can be used for the error analysis and result visualization of HOI models. It is designed originally for research but not limited to.

Currently its functionality should include:

for visualization:
- object detection result (comparison between model prediction and ground truth)
- HOI pairs for both detection result and groud truth
- HOI pairs categorized by TP/FP/False/Missed GT, for each image
- HOI pairs categorized by TP/FP/False/Missed GT, separated by verb and obj categories
- diff between the predictions of 2 models

for quantification:
- detection mAP calculation
- HOI detection mAP calculation
- AP diff between 2 models
- HOI detection error diagnosis (percentages of 7 error types)

### Development Checklist

- [x] object detection result visualization
- [x] HOI detection result visualization
- [x] HOI visualization by 4 types (False, FP, TP & Missed)
- [x] detection mAP calculation
- [x] HOI mAP calculation
- [x] **result diff/comparison visualized, between two models**
- [x] **AP diff for different classes, between 2 models**
- [x] **error factors analysis**
- [ ] code refactorization (doing, 50% progress)
- [ ] support different datasets
- [ ] code formatting
- [ ] object detection result visualization separately for each box (low priority)
- [ ] multi-processing for faster visualization

## Setup

### Prerequisites

Python3.5+ should be OK. In addition to numpy, some python libraries are also required currently. To install them, run this:
```
pip3 install loguru --user
pip3 install configargparse --user
```

You can set up this tool by simply editing config/tmp.yaml.
```yaml
# template for customization

# necessary for all:
# root dir where hico 'images' directory exists:
# also where the visualization result file goes to
root-dir: ./hico/
# path of ground truth file:
gt-path: ./hico/annotations/test_hico.json
# dir where hico category files like 'hico_list_hoi.txt' exists
hico-list-dir: ./hico/
# path of model evaluation result json file:
det-path: /data/hoi/tmp/best_predictions.json

# optional:
# image ids to show:
# should be list [start_id, end_id]
# negative indexing is supported
img-ids: [0, 10]
# img-ids: [1, 2] # for showing only image 1
# img-ids: [0, 1000000] # for showing all images (using a large value as end_id)

```

## Usage

### Supported Format

For the first version we only support the output of PPDM/Transformer (which is named best_predictions.json). The format should be:

```json
[
    {
        "file_name": "HICO_test2015_00000001.jpg", // image file name
        "predictions": [
            {
                "bbox": [10.54, 40.13, 310.21, 340.56], // [x1, y1, x2, y2]
                "category_id": 1, // from 1 to 90
                "score": 0.9483, // from 0 to 1
                // OK to add more keys
            },
            {
                ...,
            }
        ],
        "hoi_prediction": [ // only needed for HOI evaluation, not for detection evaluation
            {
                "subject_id": 0,
                "object_id": 1,
                "category_id": 88, // HOI id from 1 to 117
                "score": 0.4598, // from 0 to 1
                "binary_score": 0.98, // optional
            },
            {
                ...,
            }
        ]
    }
]

```

Similarly, the format for annotation file should be:

```json
[
    {
        "file_name": "HICO_test2015_00000001.jpg", // image file name
        "annotations": [
            {
                "bbox": [10, 40, 310, 340], // [x1, y1, x2, y2]
                "category_id": 1, // from 1 to 90
            },
            {
                ...,
            }
        ],
        "hoi_annotation": [ // only needed for HOI evaluation, not for detection evaluation
            {
                "subject_id": 0,
                "object_id": 1,
                "category_id": 88, // HOI id from 1 to 117
            },
            {
                ...,
            }
        ]
    }
]
```

### How-to-run

After setting up the tool, you can run the following command under the root directory of this repo:

```shell
# for visualization of detection result
make vis_det

# for visualization of HOI results, separated into 4 categories (False, FP, TP and Missed):
make vis_hoi

# for visualization of HOI results grouped by categories
make vis_hoi_class

# same as `vis_hoi_class`, but only for one specified verb class
# `your_class_name` can be like `adjust~`, `~tie` or `adjust~tie`
make vis_hoi_one_class class_name=your_class_name

# for the evaluation of object detection result (boxes only)
make eval_det

# for the evaluation of HOI detection mAP on HICO dataset
make eval_hico

# for error diagnosis of HOI detection result on HICO dataset
make diag_hico

# for comparison (diff) between the HOI det result of 2 models, separated into y-y, y-n, n-y, n-n:
make diff_hoi

# for comparison (diff) of Average Precision between 2 models:
make diff_ap

# for calculating and visualizing the statistics of HICO-DET dataset:
make stat_hico
```

You can also specify options via command line (instead of YAML config file), for example:

```shell
# for the evaluation of object detection result (boxes only)
python3 src/plot_detection.py --config config/tmp.yaml --img-ids 0 1000000
# plotting detection result for all test images

# for the evaluation of HOI detection mAP on hico dataset
python3 src/plot_hoi.py --config config/tmp.yaml --img-ids 0 1000000

# for the comparison between 2 models
python3 src/diff_hoi.py --config config/diff_hoi.yaml --img-ids 0 10

```

### Result

The visualization result will be saved at `$root_dir$/$model_name$`, where `$model_name$` is the name of the folder where the detection resut file exists.

## Note

This visualization tool was originally developed by [Yue Hu](https://scholar.google.com/citations?user=XBbwb78AAAAJ&hl=zh-CN) and Chi Xie for research purpose in 2021.
It is not maintained anymore.
