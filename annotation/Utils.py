#! /usr/bin/env/python3

"""
Functions and varibles used for annotation and convertion to and from CVAT annotation style
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

classes = ["recruit_live_white", "recruit_cluster_live_white", "recruit_symbiotic", "recruit_symbiotic_cluster", "recruit_partial",
           "recruit_cluster_partial", "recruit_dead", "recruit_cluster_dead", "grazer_snail", "pest_tubeworm", "unknown"]

orange = [255, 128, 0] 
blue = [0, 212, 255] 
purple = [170, 0, 255] 
yellow = [255, 255, 0] 
brown = [144, 65, 2] 
green = [0, 255, 00] 
red = [255, 0, 0]
cyan = [0, 255, 255]
dark_purple =  [128, 0, 128]
light_grey =  [192, 192, 192] 
dark_green = [0, 100, 0] 
class_colours = {classes[0]: blue,
                classes[1]: green,
                classes[2]: purple,
                classes[3]: yellow,
                classes[4]: brown,
                classes[5]: cyan,
                classes[6]: orange,
                classes[7]: red,
                classes[8]: dark_purple,
                classes[9]: light_grey,
                classes[10]: dark_green}

def binary_mask_to_rle(binary_mask):
    """binary_mask_to_rle
    Convert a binary np array into a RLE format
    
    Args:
        binary_mask (uint8 2D numpy array): binary mask

    Returns:
        rle: list of rle numbers
    """
    rle = []
    current_pixel = 0
    run_length = 0

    for pixel in binary_mask.ravel(order='C'): #go through the flaterened binary mask
        if pixel == current_pixel:  #increase number if same pixel
            run_length += 1
        else: #else save run length and reset
            rle.append(run_length) 
            run_length = 1
            current_pixel = pixel
    return rle


def poly_2_rle(points, 
               str_join: str,
               SHOW_IMAGE: bool):
    """poly_2_rle
    Converts a set of points for a polygon into an rle string and saves the data

    Args:
        points (2D numpy array): points of polygon
        str_join: how the points should be joined together
        SHOW_IMAGE (bool): True if binary mask wants to be viewed
    
    Returns:
        rle_string: string of the rle numbers,
        left: (int) left positioning of the rle numbers in pixles,
        top: (int) top positioning of the rle numbers in pixles,
        width: (int) width of the bounding box of the rle numbers in pixels,
        height: (int) height of the bounding box of the rle numbers in pixles
    """
    # create bounding box
    left = int(np.min(points[:, 0]))
    top = int(np.min(points[:, 1]))
    right = int(np.max(points[:, 0]))
    bottom = int(np.max(points[:, 1]))
    width = right - left + 1
    height = bottom - top + 1

    # create mask size of bounding box
    mask = np.zeros((height, width), dtype=np.uint8)
    # points relative to bounding box
    points[:, 0] -= left
    points[:, 1] -= top
    # fill mask where points are
    cv.fillPoly(mask, [points.astype(int)], color=1)

    # visual check of mask - looks good
    #SHOW_IMAGE = False
    if (SHOW_IMAGE):
        plt.imshow(mask, cmap='binary')
        plt.show()

    # convert the mask into a rle
    mask_rle = binary_mask_to_rle(mask)

    # rle string
    rle_string = str_join.join(map(str, mask_rle))
    
    return rle_string, left, top, width, height
