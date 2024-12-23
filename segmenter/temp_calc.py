#! /usr/bin/env python3

from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

conf_thresh = 0.01
iou_thresh = 0.6

mat_640p = np.array([
    [414, 18, 4, 4, 0, 0, 0, 0, 0, 0, 1, 91],
    [46, 218, 0, 2, 0, 0, 0, 0, 0, 0, 2, 76],
    [3, 0, 52, 1, 0, 0, 0, 0, 1, 1, 4, 23],
    [5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 2, 1, 36, 8, 1, 0, 0, 1, 0, 30],
    [0, 0, 0, 0, 14, 35, 2, 0, 0, 0, 0, 12],
    [0, 0, 0, 0, 0, 1, 6, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 3, 8],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 3],
    [42, 27, 22, 2, 42, 15, 2, 0, 3, 2, 6, 0]
])
mat_320p = np.array([
    [418, 15, 3, 5, 1, 0, 0, 0, 0, 0, 0, 87],
    [47, 225, 1, 2, 0, 0, 0, 0, 0, 0, 0, 64],
    [1, 0, 52, 0, 1, 0, 0, 0, 1, 2, 1, 11],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 38, 3, 0, 0, 0, 0, 0, 12],
    [0, 0, 0, 0, 7, 42, 3, 0, 0, 1, 0, 10],
    [0, 0, 0, 0, 0, 1, 6, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 5, 1, 5],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 7, 5],
    [42, 23, 21, 2, 44, 13, 2, 0, 3, 4, 8, 0]
])
mat_120p = np.array([
    [397,  28,   1,   5,   0,   0,   0,   0,   0,   0,   0,  92],
    [ 41, 193,   0,   1,   0,   0,   0,   0,   0,   0,   0,  31],
    [  2,   1,  54,   0,   0,   0,   0,   0,   1,   1,   1,  12],
    [  5,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   2],
    [  1,   0,   1,   1,  36,  10,   0,   0,   0,   0,   0,  14],
    [  0,   0,   0,   0,   9,  34,   1,   0,   0,   1,   0,  15],
    [  0,   0,   0,   0,   1,   1,   7,   0,   0,   0,   0,   2],
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  1,   0,   0,   0,   0,   0,   0,   0,   8,   0,   0,   1],
    [  0,   0,   3,   0,   0,   0,   0,   0,   0,   6,   4,   2],
    [  1,   0,   1,   0,   1,   0,   0,   0,   0,   1,   5,   2],
    [ 62,  41,  22,   2,  45,  14,   3,   0,   1,   3,   7,   0]
])
mat_64p = np.array([
    [145,  63,   1,   1,   0,   0,   0,   0,   1,   0,   0, 301],
    [106,  71,  17,   2,   2,   2,   0,   0,   2,   0,   3, 645],
    [ 22,  16,   7,   1,   4,   2,   0,   0,   1,   3,   2, 312],
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  1,   1,   4,   0,  13,  10,   1,   0,   0,   1,   0, 107],
    [  0,   2,   0,   0,   9,  13,   2,   0,   1,   1,   0,  99],
    [  0,   1,   0,   0,   7,  11,   3,   0,   0,   0,   0,  57],
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   1,   7,   0,   0,   0,   0,   0,   0,   0,   1,  15],
    [ 18,  14,  21,   0,   2,   1,   1,   0,   1,   1,   3, 133],
    [218,  94,  25,   6,  55,  20,   4,   0,   4,   6,   8,   0]
])
mat_480 = np.array([
    [428,  23,   2,   5,   1,   0,   0,   0,   0,   0,   0,  72],
    [ 34, 214,   0,   1,   0,   0,   0,   0,   0,   0,   2,  61],
    [  3,   0,  52,   2,   0,   0,   0,   0,   1,   0,   2,  12],
    [  0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  1,   0,   2,   1,  33,   7,   0,   0,   0,   0,   0,  14],
    [  0,   0,   0,   0,  11,  31,   1,   0,   0,   0,   0,   6],
    [  0,   0,   0,   0,   0,   1,   7,   0,   0,   0,   0,   2],
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2],
    [  0,   0,   0,   0,   0,   0,   0,   0,   7,   0,   0,   0],
    [  0,   0,   3,   0,   0,   0,   0,   0,   0,   5,   1,   3],
    [  0,   1,   0,   0,   0,   0,   0,   0,   0,   1,   5,   5],
    [ 44,  25,  22,   1,  47,  20,   3,   0,   2,   6,   7,   0]
])

test_320p_2 =np.array([
   [427,  25,   2,   4,   0,   0,   0,   0,   0,   0,   1,  92],
    [ 31, 213,   0,   3,   0,   0,   0,   0,   0,   0,   0,  42],
    [  1,   0,  53,   0,   0,   0,   0,   0,   0,   1,   1,  17],
    [  3,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   2],
    [  0,   0,   0,   0,  30,   8,   0,   0,   0,   0,   0,  15],
    [  0,   0,   0,   0,  16,  41,   1,   0,   0,   1,   0,  10],
    [  0,   0,   0,   0,   0,   0,   7,   0,   0,   0,   0,   1],
    [  0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   1],
    [  0,   0,   0,   0,   0,   0,   0,   0,   6,   0,   0,   1],
    [  1,   0,   6,   0,   0,   0,   0,   0,   0,   6,   2,   4],
    [  1,   0,   1,   1,   0,   0,   0,   0,   0,   1,   7,   2],
    [ 46,  25,  19,   2,  46,  10,   3,   0,   3,   3,   6,   0]
])


conf_mat_d = test_320p_2

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

    TPR = np.where((tp_d+ fn_d) > 0, tp_d/ (tp_d+ fn_d), 0)[mask]
    FNR = np.where((tp_d+ fn_d) > 0, fn_d / (tp_d+ fn_d), 0)[mask]
    FPR = np.where((fp_d+ tn_d) > 0, fp_d/ (fp_d+ tn_d), 0)[mask]
    TNR = np.where((fp_d+ tn_d) > 0, tn_d / (fp_d+ tn_d), 0)[mask]
    TPmean = np.mean(TPR)
    FNmean = np.mean(FNR)
    FPmean = np.mean(FPR)
    TNmean = np.mean(TNR)
    return TPmean, FNmean, FPmean, TNmean

def p_r_f1(conf_mat, class_ignore=None):
    if class_ignore is None:
        class_ignore = []
    
    mask = np.ones(conf_mat.shape[0], dtype=bool)
    mask[class_ignore] = False

    tp_d = conf_mat.diagonal()
    fp_d = conf_mat.sum(1) - tp_d
    fn_d = conf_mat.sum(0) - tp_d

    precision = np.where((tp_d + fp_d) > 0, tp_d / (tp_d + fp_d), 0)[mask]
    recall = np.where((tp_d + fn_d) > 0, tp_d / (tp_d + fn_d), 0)[mask]
    F1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)

    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_F1 = np.mean(F1)

    return avg_precision, avg_recall, avg_F1

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

TPmean, FNmean, FPmean, TNmean = get_TP_FP_FN_TN(conf_mat_d)
data = np.array([[TPmean, TNmean], [FPmean, FNmean]])
plot_results(data, conf_thresh, iou_thresh)


precision, recall, F1 = p_r_f1(conf_mat_d)

print("---------------Results---------------------")

print(f"TPmean={TPmean:.4f}, FPmean={FPmean:.4f}, FNmean={FNmean:.4f}, TNmean={TNmean:.4f}")
print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={F1:.4f}")

print("Done")
import code
code.interact(local=dict(globals(), **locals()))

