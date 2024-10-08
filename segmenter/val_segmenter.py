#! /usr/bin/env python3

"""val_segmenter.py
validate a model against a dataset
"""
#TODO: Rates of TP, FP, FN, TN
#TODO: no of images in datset, info on what dataset is.
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

weights_file = '/home/java/Java/ultralytics/runs/segment/train21/weights/best.pt' #model
conf_thresh = 0.01
iou_thresh = 0.6


def get_TP_FP_FN_TN(conf_mat, class_ignore=None):
    """get_TP_FP_FN_TN
        Get the average True Positive, False Positive, False Negative and True Negative 
        rates for the confusion matrix ignoring the classes in class_ignore.
    """
    if class_ignore is None:
        class_ignore = []
    
    tp_d = conf_mat.diagonal()
    fp_d = conf_mat.sum(1) - tp_d
    fn_d = conf_mat.sum(0) - tp_d
    total_samples = conf_mat.sum()
    tn_d = np.zeros(conf_mat.shape[0])
    for i in range(conf_mat.shape[0]):
        tp = conf_mat[i, i]
        row_sum = conf_mat[i, :].sum()
        col_sum = conf_mat[:, i].sum()
        tn_d[i] = total_samples - row_sum - col_sum + tp

    mask = np.ones(conf_mat.shape[0], dtype=bool)
    mask[class_ignore] = False

    TPR = np.where((tp_d+ fn_d) != 0, tp_d/ (tp_d+ fn_d), 0)[mask]
    FNR = np.where((tp_d+ fn_d) != 0, fn_d / (tp_d+ fn_d), 0)[mask]
    FPR = np.where((fp_d+ tn_d) != 0, fp_d/ (fp_d+ tn_d), 0)[mask]
    TNR = np.where((fp_d+ tn_d) != 0, tn_d / (fp_d+ tn_d), 0)[mask]
    TPmean = np.mean(TPR)
    FNmean = np.mean(FNR)
    FPmean = np.mean(FPR)
    TNmean = np.mean(TNR)
    return TPmean, FNmean, FPmean, TNmean

def plot_results(data, conf_thresh, iou_thresh):
    """plot_results
    Plot the data in a heatmap
    """
    x_labels = ['P (Positive)', 'N (Negative)']  # x-axis labels
    y_labels = ['T (True)', 'F (False)']  # y-axis labels
    plt.figure(figsize=(6, 4))
    sns.heatmap(data, annot=True, fmt=".2%", cmap='Blues', xticklabels=x_labels, yticklabels=y_labels)
    plt.title(f"Simplified confusion matrix with confidence of {conf_thresh:.2f} and IOU of {iou_thresh:.2f}")
    plt.show()


# Load a model
model = YOLO(weights_file)

## Validate the model
#if not arguments, will use the defult arguments and validate on the Val files of the model trained dataset.
metrics_d = model.val(plots=True) #DEFAULT, conf=0.001, iou=0.6 #data='/home/java/Java/Cgras/cgras_settler_counter/segmenter/cgras_20230421.yaml',
f1_d = metrics_d.box.f1  # F1 score for each class
tp_d, fp_d = metrics_d.confusion_matrix.tp_fp() # returns 2 arrays, 1xN where N is the number of classes.
conf_mat_d = metrics_d.confusion_matrix.matrix #has the confusion matrix as NXN array
conf_mat_normalised = conf_mat_d / (conf_mat_d.sum(0).reshape(1, -1) + 1E-9)

metrics_max = model.val(conf=max(f1_d), iou=iou_thresh, plots=True)
conf_mat_max = metrics_max.confusion_matrix.matrix

TPmean, FNmean, FPmean, TNmean = get_TP_FP_FN_TN(conf_mat_d, class_ignore=[8, 9, 10, 11])
data = np.array([[TPmean, TNmean], [FPmean, FNmean]])
plot_results(data, conf_thresh, iou_thresh)

TPmean, FNmean, FPmean, TNmean = get_TP_FP_FN_TN(conf_mat_max, class_ignore=[8, 9, 10, 11])
data = np.array([[TPmean, TNmean], [FPmean, FNmean]])
plot_results(data, max(f1_d), iou_thresh)

## Visulise instances


print("Done")
import code
code.interact(local=dict(globals(), **locals()))


metrics_d.box.map    # map50-95
metrics_d.box.map50  # map50
metrics_d.box.map75  # map75
metrics_d.box.maps   # a list contains map50-95 of each category
metrics_d.box.p   # Precision for each class
metrics_d.box.r   # Recall for each class
metrics_d.box.all_ap  # AP scores for all classes and all IoU thresholds
metrics_d.box.ap_class_index  # Index of class for each AP
metrics_d.box.nc  # Number of classes
metrics_d.box.ap50  # AP at IoU threshold of 0.5 for all classes
metrics_d.box.ap  # AP at IoU thresholds from 0.5 to 0.95 for all classes
metrics_d.box.mp  # Mean precision of all classes
metrics_d.box.mr  # Mean recall of all classes
metrics_d.box.map50  # Mean AP at IoU threshold of 0.5 for all classes
metrics_d.box.map75  # Mean AP at IoU threshold of 0.75 for all classes
metrics_d.box.map  # Mean AP at IoU thresholds from 0.5 to 0.95 for all classes
metrics_d.box.mean_results  # Mean of results
metrics_d.box.class_result  # Class-aware result
metrics_d.box.maps  # mAP of each class
metrics_d.box.fitness  # Model fitness as a weighted combination of metrics_d
metrics_d.confusion_matrix.print() #print RAW confusion metrix
