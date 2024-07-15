#! /usr/bin/env python3

""" min_res.py
script to run the experiment of minimum resolution for coral detection model
    Using a neural network (eg, yolov8-seg), what is the minimum resolution required for it to detect a coral?

    Assumptions:

    Appropriately trained NN - eg, best we can get, say.. 1000 representative image

    Testing on representative images/different classes, all approximately the same real-world size with sufficient detail for detection

    Same physical working distance/camera sensor across the images - camera pose does not change (we'll just be using the same set of images)

    Same network, trained for the fixed, appropriate image size

    Assume starting point (highest resolution) is suffucient (it is)

    Centre of the image - minimal distortion - will need to do a similar experiment at the edges of the image!

    We are training on unrectified images, therefore providing the NN unrectified images
    
"""
import matplotlib.pyplot as plt
import cv2 as cv
import os
from ultralytics import YOLO
import torch
import supervision as sv


print('min_res.py')

# output directory
out_dir = '/home/dorian/Code/cgras_ws/cgras_settler_counter/resolution_experiment/output/'
os.makedirs(out_dir, exist_ok=True)

# load NN 
weight_file = '/home/dorian/Code/cgras_ws/cgras_settler_counter/resolution_experiment/model/cgras_yolov8n-seg_640p_20231209.pt'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weight_file).to(device)

# mask annotator for plotting
mask_annotator = sv.MaskAnnotator()


###############################

# location of image
image_name = '/home/dorian/Code/cgras_ws/cgras_settler_counter/resolution_experiment/images/20230105_MIS1_RC_Aspat_T04_08_1280crop.jpg'
base_name = os.path.basename(image_name)[:-4]

# load image, get image dimensions
image = cv.imread(image_name)
height, width, chan = image.shape
ar = width/height # aspect ratio

# import annotation(s) in yolov8 format for a mask
label = '/home/dorian/Code/cgras_ws/cgras_settler_counter/resolution_experiment/labels/20230105_MIS1_RC_Aspat_T04_08_1280crop.txt'
# NOTE: need to deal with cvat export issues


# decide image resolution parameters
# image resolution array for the largest dimension, needs to be a multiple of 32
image_sizes = [1280, 640, 480, 320, 160]


# loop over image sizes, resize and detect on each
image_array = []
for i in image_sizes:
    
    # resize image
    width_r = i
    height_r = round(width_r / ar)
    image_r = cv.resize(image, (width_r, height_r), interpolation=cv.INTER_LINEAR) # resized image
    image_array.append(image_r)
    
    # show image
    # cv.imshow('resize: {width_r}', image_r)
    # cv.waitKey()
    
    # save image
    save_file = out_dir+base_name+'_'+str(width_r)+'.jpg'
    cv.imwrite(save_file,image_r)
    
    # do inference on images
    results = model(image_r) # need to specify input image size?
    
    # show detections on image
    detections = sv.Detections.from_ultralytics(results[0])
    annotated_image = mask_annotator.annotate(scene=image_r, detections=detections)
    cv.imwrite(f"{os.path.join(out_dir,base_name)}_det_{width_r}.jpg", annotated_image)
    
    # get metric(s) on image (IoU)


# generate plot of metric vs res

# NOTE: from 2022 dataset (100 images, all images used for training)
import code
code.interact(local=dict(globals(), **locals()))
