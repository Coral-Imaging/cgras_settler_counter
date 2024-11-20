#! /usr/bin/env/python3

"""
Functions and varibles used for annotation and convertion to and from CVAT annotation style
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

classes = ["recruit_live_white", "recruit_cluster_live_white", "recruit_symbiotic", "recruit_cluster_symbiotic", "recruit_partial",
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
    """
    Convert a binary np array into a RLE format.
    Args:
        binary_mask (uint8 2D numpy array): binary mask
    Returns:
        rle: list of rle numbers
    """
    # Flatten the binary mask and append a zero at the end to handle edge case
    flat_mask = binary_mask.flatten()
    # Find the positions where the values change
    changes = np.diff(flat_mask)
    # Get the run lengths
    runs = np.where(changes != 0)[0] + 1
    # Get the lengths of each run
    run_lengths = np.diff(np.concatenate([[0], runs]))
    return run_lengths.tolist()

def rle_to_binary_mask(rle_list, 
                       width: int, 
                       height: int, 
                       SHOW_IMAGE: bool):
    """rle_to_binary_mask
    Converts a rle_list into a binary np array. Used to check the binary_mask_to_rle function

    Args:
        rle_list (list of strings): containing the rle information
        width (int): width of shape
        height (int): height of shape
        SHOW_IMAGE (bool): True if binary mask wants to be viewed

    Returns:
        mask: uint8 2D np array
    """
    mask = np.zeros((height, width), dtype=np.uint8) 
    current_pixel = 0
    
    for i in range(0, len(rle_list)):
        run_length = int(rle_list[i]) #find the length of current 0 or 1 run
        if (i % 2 == 0): #if an even number the pixel value will be 0
            run_value = 0
        else:
            run_value = 1

        for j in range(run_length): #fill the pixel with the correct value
            mask.flat[current_pixel] = run_value 
            current_pixel += 1

    if (SHOW_IMAGE):
        print("rle_list to binary mask")
        plt.imshow(mask, cmap='binary')
        plt.show()

    return mask


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
