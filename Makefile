# to use this visualization tool, 
# please customize the config in config/tmp.yaml


# ================== Visualization =======================
# Visualize the detection boxes and HOI pairs in each image.
# You can specify the image ids.
vis_det:
	python3 src/plot_detection.py --config config/tmp.yaml

# Visualize the HOI pairs (FP, TP, False, Missed) in each image.
# You can specify the image ids.
vis_hoi:
	python3 src/plot_hoi.py --config config/tmp.yaml

# Visualize the HOI detection differences (YY,YN,NY,NN,unmatched) of 2 models.
diff_hoi:
	python3 src/diff_hoi.py --config config/diff_hoi.yaml

# ============ HICO only =============
# Visualize the HOI pairs (FP, TP, False, Missed) by verb and object types.
vis_hoi_class:
	rlaunch --cpu=8 --gpu=0 --memory=10240 -- python3 src/plot_hoi_classwise.py --config config/tmp.yaml

# You can run `make vis_hoi_one_class class_name=adjust~tie`
# or `make vis_hoi_one_class class_name=adjust~`
# or `make vis_hoi_one_class class_name=~tie`
vis_hoi_one_class:
	rlaunch --cpu=16 --gpu=0 --memory=10240 -- python3 src/plot_hoi_classwise.py --config config/tmp.yaml --class-name ${class_name}
# if you want to specifiy range for plotting,
# you can use additional param --score-range:
	# rlaunch --cpu=16 --gpu=0 --memory=10240 -- python3 src/plot_hoi.py --config config/tmp.yaml --classwise-plot --class-name ${class_name} --score-range 0.1 0.5

vis_hoi_top_class:
	rlaunch --cpu=16 --gpu=0 --memory=10240 -- python3 src/plot_hoi_classwise_top.py --config config/tmp.yaml --topk-cls-for-vis 0 10

# ============ VCOCO only =============
# Visualize the HOI pairs (FP, TP, False, Missed) by verb and object types.
vis_hoi_class_vcoco:
	rlaunch --cpu=8 --gpu=0 --memory=10240 -- python3 src/plot_hoi_classwise_vcoco.py --config config/tmp.yaml --classwise-plot

vis_hoi_one_class_vcoco:
	rlaunch --cpu=16 --gpu=0 --memory=10240 -- python3 src/plot_hoi_classwise_vcoco.py --config config/tmp.yaml --classwise-plot --class-name ${class_name}





# ================== Statistics =======================
# Perform detection mAP evaluation.
eval_det:
	python3 src/eval_detection.py --config config/tmp.yaml

# Diff the mAP of 2 models.
diff_ap:
	python3 src/eval_hico.py --config config/diff_hoi.yaml --diff-ap

# ============ HICO only =============
# Perform HOI mAP evaluation on HICO dataset.
eval_hico:
	python3 src/eval_hico.py --config config/tmp.yaml

# Perform HOI error diagnosis on HICO dataset.
diag_hico:
	python3 src/eval_hico.py --config config/tmp.yaml --diag

# Analyze HICO data statistics.
stat_hico:
	python3 src/stat_hico.py --config config/stat.yaml
