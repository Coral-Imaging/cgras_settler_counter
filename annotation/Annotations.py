#! /usr/bin/env/python3

"""
annotation class to segment and create CVAT annotations
"""

import os
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
import xml.etree.ElementTree as ET



class Annotation:
    
    SAM_MODEL_DEFAULT = '/home/dorian/Code/cgras_ws/segment-anything/model/sam_vit_h_4b8939.pth'
    LABEL_DEFAULT = 1
    CLASS_LABEL_DEFAULT = 'recruit_live_white'
    
    def __init__(self,
                 input_annotation_file: str,
                 image_dir: str,
                 output_dir: str,
                 sam_model: str = SAM_MODEL_DEFAULT,
                 scale: float = 0.25,
                 label_default: int = LABEL_DEFAULT,
                 class_label_default: str = CLASS_LABEL_DEFAULT) -> None:
        
        self.input_annotation_file = input_annotation_file
        self.image_dir = image_dir
        self.output_dir = output_dir
        
        self.annotations = []
        
        # could also get list of images from annotations file
        # TODO write a function to ensure images are consistent between the two sources
        self.images_from_dir = sorted(os.listdir(image_dir))
        
        # setup Segment Anything Model
        self.sam = sam_model_registry['vit_h'](checkpoint=sam_model)
        self.sam.to(device='cuda')
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        
        self.scale = scale # TODO simply resize images to 1024
        
        self.label_default = label_default
        self.class_label_default = class_label_default
        
        # plot settings
        # percent as a function of max image shape
        self.line_thickness_percent = 0.002
        self.font_thickness_percent = 0.00045
        self.font_scale_percent = 0.0005
        self.lines_color = [255, 0, 0] # RGB
        
        pass
    
    
    def get_line_thickness_from_image(self, image):
        """ get line/font thickness numbers for a given image """
        # run per image, because image size may change
        
        # image plotting settings
        lines_thick = int(np.ceil(self.line_thickness_percent* max(image.shape)))
        font_scale = max(1, self.font_scale_percent * max(image.shape))
        font_thick = int(np.ceil(self.font_thickness_percent * max(image.shape)))
        return {'lines_thick': lines_thick, 'font_scale': font_scale, 'font_thick': font_thick}
    
    
    def is_within_percent(a, b, percent):
        # determine if a is within a specified percentage of b, returning True
        # otherwise False
        threshold = abs(b * percent / 100.0)
        return abs(a - b) <= threshold


    def show_anns(anns):
        """ function from SAM ipynotebook to show annotations (masks) on an image """
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, m*0.35)))
        
        
    def show_masks(self, image, masks):
        """ show masks """
        # code to show the whole image with masks ontop     
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        self.show_anns(masks)
        plt.axis('off')
        plt.show() 
    
    
    def generate_masks(self, image_files):
        """ generate  sam to get masks from a given list of images, images in image_dir """ 
        
        # iterate over the image files
        # MAX_IMG = 2
        polygons_all = []
        class_labels_all = []
        
        for i, image_name in enumerate(image_files):
            
            # open/read image
            image = cv.imread(os.path.join(self.image_dir, image_name))
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            
            # get image plotting scales/thicknesses
            plot_dict = self.get_line_thickness_from_image(image)
            
            # downsize image
            image_r = cv.resize(image, (0, 0), fx=self.scale, fy=self.scale)
            
            # masks from SAM
            masks = self.mask_generator.generate(image_r)
            # self.show_masks(image_r, masks)
            
            # Mask generation returns a list over masks, where each mask is a
            # dictionary containing various data about the mask. These keys are:
            # * `segmentation` : the mask
            # * `area` : the area of the mask in pixels
            # * `bbox` : the boundary box of the mask in XYWH format
            # * `predicted_iou` : the model's own prediction for the quality of the mask
            # * `point_coords` : the sampled input point that generated this mask
            # * `stability_score` : an additional measure of mask quality
            # * `crop_box` : the crop of the image used to generate this mask in XYWH format
            
            # sort masks wrt area in reverse order (largest first)
            sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
            
            # remove background polygon - hack wrt image size
            # largest_mask = sorted_masks[0]
            # if largest_mask['bbox'][2] is within 80% of image_width and Y is within 80% of image_height
            image_height, image_width, _ = image.shape
            largest_mask = sorted_masks[0]
            largest_mask_width = largest_mask['bbox'][2] / self.scale
            largest_mask_height = largest_mask['bbox'][3] / self.scale
            if self.is_within_percent(largest_mask_width, image_width, 80) and \
                self.is_within_percent(largest_mask_height, image_height, 80):
                # largest mask is basically just image boundary/background, so remove
                del sorted_masks[0]


            
            
            
            