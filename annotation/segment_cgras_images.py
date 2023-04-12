#! /usr/bin/env/python3

""" test segment_anything """

# following the segment_anything github 
# https://github.com/facebookresearch/segment-anything

# downloaded model checkpoints
# https://github.com/facebookresearch/segment-anything#model-checkpoints

import os
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
# from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

# sample image
img_dir = '/home/dorian/Data/cgras_dataset_20230403_small'
img_files = sorted(os.listdir(img_dir))
image = cv.imread(os.path.join(img_dir, img_files[0]))
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
print(f'img size = {image.shape}')


# resize image
scale = 0.5
image = cv.resize(image, (0,0), fx=scale, fy=scale)
print(f'resize img size = {image.shape}')

# segment anything
sam = sam_model_registry['vit_h'](checkpoint='/home/dorian/Code/cgras_ws/segment-anything/model/sam_vit_h_4b8939.pth')
sam.to(device='cuda')
# predictor = SamPredictor(sam)
# predictor.set_image(img)
# masks, _, _= predictor.predict('prompt')

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
# * `segmentation` : the mask
# * `area` : the area of the mask in pixels
# * `bbox` : the boundary box of the mask in XYWH format
# * `predicted_iou` : the model's own prediction for the quality of the mask
# * `point_coords` : the sampled input point that generated this mask
# * `stability_score` : an additional measure of mask quality
# * `crop_box` : the crop of the image used to generate this mask in XYWH format

print(len(masks))
print(masks[0].keys())

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show() 

import code
code.interact(local=dict(globals(), **locals()))