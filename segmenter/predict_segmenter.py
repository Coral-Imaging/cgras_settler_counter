#! /usr/bin/env python3

""" predict_segmenter.py
run segmentation model on folder of images
"""

from ultralytics import YOLO
import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt

# load model
weights_file = '/home/dorian/Code/cgras_ws/cgras_settler_counter/segmenter/weights/20240923_tiledimages_yolov8xseg_naive.pt' #model
# load custom model (YOLOv8)
model = YOLO(weights_file)

# output directory (for plots/detections)
out_dir = '/home/dorian/Data/cgras_data_2023_tiled/predict/train'
os.makedirs(out_dir, exist_ok=True)

# image directory
# TODO first run on training images to make sure network is getting right training data
# TODO then run on val/test sets
img_dir = '/home/dorian/Data/cgras_data_2023_tiled/split/train/images'
img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))


# TODO should import labels and plot labels for image-by-image comparison


print(f'Length of img_list = {len(img_list)}')

for i, img_name in enumerate(img_list):
    print(f'{i}/{len(img_list)}: {os.path.basename(img_name)}')
    # model inference
    results = model(img_name, 
                    save=True, 
                    save_txt=True, 
                    save_conf=True, 
                    boxes=True,
                    conf=0.3, # intentionally very low for debugging purposes
                    agnostic_nms=True)
    
    res_plotted = results[0].plot()
    res_rgb = cv.cvtColor(res_plotted, cv.COLOR_BGR2RGB)
    # save image
    img_save_name = os.path.basename(img_name).rsplit('.')[0] + '_box.jpg'
    plt.imsave(os.path.join(out_dir, img_save_name), res_rgb)
    # plt.show()
    