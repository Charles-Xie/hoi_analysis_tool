#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# argument settings, can be customized by using a different YAML config file
# or passing argument values directly via command line

import configargparse


class Range(object):
    def __init__(self, start, end):
        r"""Restrict argparse param value to a range.
        """
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end


parser = configargparse.ArgParser(default_config_files=[])
parser.add("--config", required=True, is_config_file=True, help="config file path")
# basic options
parser.add("--root-dir", required=True, type=str, help="root dir for hico dataset images and annotations")
parser.add("--gt-path", required=True, type=str, help="path for gt json file of HICO dataset (like 'train_hico.json' or 'test_hico.json')")
parser.add("--det-path", required=True, type=str, help="path for HOI evaluation result json file (like 'best_predictions.json')")
parser.add("--hico-list-dir", required=True, type=str, help="dir where hico category files like 'hico_list_hoi.txt' exists")
parser.add("--img-ids", type=int, default=[0, 10], nargs='+', metavar="--img-ids 0 10", help="image ids [start, end] or[id1, id2, id2, ..., ] selected for visualization")

# additional options for some features
# for making comparisons between 2 models, like `diff_hoi` and `diff_ap`
parser.add("--det2-path", type=str, required=False, help="path for second model's HOI evaluation result, only needed when diff 2 models")

# for `diff_ap`, whether or not
parser.add("--diff-ap", action="store_true", default=False, help="whether or not to diff ap for 2 models, only needed for diff_ap")

# for `vis_det`
parser.add("--det-vis-thres", required=False, type=float, choices=[Range(0.0, 1.0)], help="threshold for the visualization of det boxes")

# for `diag_hico`, whether or not
parser.add("--diag", action="store_true", default=False, help="whether or not to diagnose model errors, only needed for analyze_error")

# for `vis_hoi_class`. if not, `vis_hoi`
parser.add("--class-name", type=str, required=False, help="optional, name of a specific class selected for vis, like 'adjust~', '~tie' or 'adjust~tie'")
parser.add("--score-range", type=float, default=[-1000, 1000], nargs=2, metavar="--score-range -1000 1000", help="verb score range [lower, upper] for visualizing False and FP pairs")
parser.add("--topk-cls-for-vis", type=int, default=[0, 10], nargs=2, metavar="--topk-cls-for-vis 0 10", help="top AP class numbers [start, end] for `vis_hoi_top_class`")

args = parser.parse_args()

# conflicts
assert not (args.diff_ap and args.diag), "You should not set --diag "\
    "and --diff-ap at the same time"
print("Arguments:")
print(args)
