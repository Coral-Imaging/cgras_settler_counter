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
from copy import deepcopy

# website/blog that finally describes the yolov5/8 format for masks as bounding polygons
# https://towardsdatascience.com/trian-yolov8-instance-segmentation-on-your-data-6ffa04b2debd
# class# <x1 y1 x2 y2 ...>
#  where x y coordinates are normalised to 1 wrt image width and height, respectively
def get_bounding_polygon(mask_binary):
        """get_bounding_polygon
        Finds the contour surrounding the blobs within the binary mask

        Args:
            mask_binary (uint8 2D numpy array): binary mask

        Returns:
            all_x, all_y: all the x, y points of the contours as lists
        """        
        # convert to correct type for findcontours
        if not mask_binary.dtype == np.uint8:
            mask_binary = mask_binary.astype(np.uint8) 

        # find bounding polygon of binary image
        contours_in, hierarchy = cv.findContours(mask_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = list(contours_in)
        # we take the largest contour as the most likely polygon from the mask
        contours.sort(key=len, reverse=True)
        largest_contour = contours[0]
        largest_contour_squeeze = np.squeeze(largest_contour)
        all_x, all_y = [], []
        for c in largest_contour_squeeze:
            all_x.append(c[0])
            all_y.append(c[1])

        return all_x, all_y


def plot_poly(image,
              poly,
              lines_thick,
              font_scale,
              lines_color,
              font_thick):
        """plot_poly
        plot detection polygon onto image

        Args:
            image (_type_): _description_
            detection (_type_): _description_
            lines_thick (_type_): _description_
            font_scale (_type_): _description_
            lines_color (_type_): _description_
            font_thick (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # reshape polygon from (x, y) array lists to vertical for polylines 
        poly = poly.transpose().reshape((-1, 1, 2))
        # draw polygons onto image
        cv.polylines(image,
                     pts=np.int32([poly]),
                     isClosed=True,
                     color=lines_color,
                     thickness=lines_thick,
                     lineType=cv.LINE_4)

        # find a place to put polygon class prediction and confidence we find
        # the x,y-pair that's closest to the top-left corner of the image, but
        # also need it to be ON the polygon, so we know which one it is and thus
        # most likely to actually be in the image
        # distances = np.linalg.norm(detection.poly, axis=0)
        # minidx = np.argmin(distances)
        # xmin = detection.poly[0, minidx]
        # ymin = detection.poly[1, minidx]

        # add text to top left corner of box
        # class + confidence as a percent
        # conf_str = format(detection.score * 100.0, '.0f')
        # detection_str = '{}: {}'.format(detection.class_name, conf_str) 
        
        # image = draw_rectangle_with_text(image, 
        #                                 text=detection_str,
        #                                 xy=(xmin, ymin), 
        #                                 font_scale=font_scale, 
        #                                 font_thickness=font_thick, 
        #                                 rect_color=lines_color)
        return image



def save_image(image, image_filename: str):
    """save_image
    write image to file, given image and image_filename, assume image in in
    RGB format

    Args:
        image (_type_): _description_
        image_filename (str): _description_

    Returns:
        _type_: _description_
    """     
    # make sure directory exists
    image_dir = os.path.dirname(image_filename)
    os.makedirs(image_dir, exist_ok=True)

    # assuming image is in RGB format, so convert back to BGR
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(image_filename, image)
    return True


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


# determine if a is within a specified percentage of b, returning True otherwise
# False
def is_within_percent(a, b, percent):
     threshold = abs(b * percent / 100.0)
     return abs(a - b) <= threshold




########################################################################################################################
# segment anything
sam = sam_model_registry['vit_h'](checkpoint='/home/dorian/Code/cgras_ws/segment-anything/model/sam_vit_h_4b8939.pth')
sam.to(device='cuda')
# predictor = SamPredictor(sam)
# predictor.set_image(img)
# masks, _, _= predictor.predict('prompt')

mask_generator = SamAutomaticMaskGenerator(sam)


# image rescale/down-size requirements
scale = 0.5 # the max I can do without running into space errors currently, need to recale the annotations

# sample image
img_dir = '/home/dorian/Data/cgras_dataset_20230403_small'
img_files = sorted(os.listdir(img_dir))

# output directory
out_dir = '/home/dorian/Data/cgras_dataset_20230403_small_poly'
os.makedirs(out_dir, exist_ok=True)


for i, image_name in enumerate(img_files):
    print(f'{i+1}/{len(img_files)}: masking {image_name}')
     
    image = cv.imread(os.path.join(img_dir, image_name))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # image_width, image_height = image.shape()
    # print(f'img size = {image.shape}')
    
    # image plotting settings
    lines_thick = int(np.ceil(0.002 * max(image.shape)))
    font_scale = max(1, 0.0005 * max(image.shape))
    lines_color = [255, 0, 0]
    font_thick = int(np.ceil(0.00045 * max(image.shape)))

    # downsize image
    image_r = cv.resize(image, (0,0), fx=scale, fy=scale)
    # print(f'resize img size = {image_r.shape}')

    masks = mask_generator.generate(image_r)

    # Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
    # * `segmentation` : the mask
    # * `area` : the area of the mask in pixels
    # * `bbox` : the boundary box of the mask in XYWH format
    # * `predicted_iou` : the model's own prediction for the quality of the mask
    # * `point_coords` : the sampled input point that generated this mask
    # * `stability_score` : an additional measure of mask quality
    # * `crop_box` : the crop of the image used to generate this mask in XYWH format

    # print(len(masks))
    # print(masks[0].keys())

    # sort masks wrt area in reverse order (largest first)
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    # code to show the whole image with masks ontop     
    # plt.figure(figsize=(20,20))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show() 

    # remove background polygon - hack wrt image size
    # largest_mask = sorted_masks[0]
    # if largest_mask['bbox'][2] is within 80% of image_width and Y is within 80% of image_height
    
    image_width, image_height, _ = image.shape
    largest_mask = sorted_masks[0]
    largest_mask_width = largest_mask['bbox'][2] / scale
    largest_mask_height = largest_mask['bbox'][3] / scale
    if is_within_percent(largest_mask_width, image_width, 80) and is_within_percent(largest_mask_height, image_height, 80):
         # largest mask is basically just image boundary/background, so remove
        del sorted_masks[0]

    # get bounding polygons for each
    polygons_unscaled = []
    for mask in sorted_masks:
        mask_bin = mask['segmentation'].astype(np.uint8)
        x, y = get_bounding_polygon(mask_bin)
        poly = np.array((x, y))
        mask['poly'] = poly
        # rescale to original image size
        polygons_unscaled.append(poly / scale)

    # plot polygons over original image scale
    # plot polygons onto image
    image_p = deepcopy(image)

    for i, p in enumerate(polygons_unscaled):
        # if i == 0:
        #      # try skipping the first one?
        #      continue
        # else:
        plot_poly(image_p, p, lines_thick, font_scale, lines_color, font_thick)
    

    # save image
    save_img_name = os.path.splitext(image_name)[0] + '_polygons.png'
    save_image(image_p, os.path.join(out_dir, save_img_name))

    # TODO convert polygons into YOLO format
    # TODO save as .txt file to upload to CVAT
    
print('Done')


import code
code.interact(local=dict(globals(), **locals()))