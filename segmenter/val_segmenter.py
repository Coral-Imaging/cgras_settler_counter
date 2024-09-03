#! /usr/bin/env python3

"""val_segmenter.py
validate a model against a dataset
"""
#TODO: Rates of TP, FP, FN, TN
#TODO: no of images in datset, info on what dataset is.
from ultralytics import YOLO

weights_file = '/home/java/Java/ultralytics/runs/segment/train16/weights/best.pt' #model

# Load a model
model = YOLO(weights_file)

# Validate the model
#if not arguments, will use the defult arguments and validate on the Val files of the model trained dataset.
metrics = model.val(data='/home/java/Java/Cgras/cgras_settler_counter/segmenter/cgras_20230421.yaml', conf=0.001, iou=0.6) 

metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
metrics.box.p   # Precision for each class
metrics.box.r   # Recall for each class
metrics.box.f1  # F1 score for each class
metrics.box.all_ap  # AP scores for all classes and all IoU thresholds
metrics.box.ap_class_index  # Index of class for each AP
metrics.box.nc  # Number of classes
metrics.box.ap50  # AP at IoU threshold of 0.5 for all classes
metrics.box.ap  # AP at IoU thresholds from 0.5 to 0.95 for all classes
metrics.box.mp  # Mean precision of all classes
metrics.box.mr  # Mean recall of all classes
metrics.box.map50  # Mean AP at IoU threshold of 0.5 for all classes
metrics.box.map75  # Mean AP at IoU threshold of 0.75 for all classes
metrics.box.map  # Mean AP at IoU thresholds from 0.5 to 0.95 for all classes
metrics.box.mean_results  # Mean of results
metrics.box.class_result  # Class-aware result
metrics.box.maps  # mAP of each class
metrics.box.fitness  # Model fitness as a weighted combination of metrics

print("Done")
import code
code.interact(local=dict(globals(), **locals()))

