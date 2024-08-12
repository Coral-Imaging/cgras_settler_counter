#! /usr/bin/env python3

""" pde_to_yolo.py
    Convert PDE annotations from CSV files to YOLO format
"""
import os
import pandas as pd
import numpy as np
import cv2 as cv
import glob
import re
import math
from Utils import poly_2_rle

#segment_filename = f"{os.path.splitext(filename)[0]}_segment_{left}_{upper}_{right}_{lower}.jpg"
img_path= '/media/java/cslics_ssd1/SCU_Pdae_Data/T3jpg'
label_path = '/media/java/cslics_ssd1/SCU_Pdae_Data/ROIs Extracted/T03 (Done)'
imgsave_path = '/media/java/cslics_ssd1/SCU_Pdae_Data/annotated'
img_list = sorted(glob.glob(os.path.join(img_path, '*.jpg')))

def contained_within(box1, box2):
    """Check if box1 is contained within box2"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return x1 >= x3 and y1 >= y3 and x2 <= x4 and y2 <= y4

for i, img in enumerate(img_list):
    print(f'Processing {i} of {len(img_list)}: {img}')

    image = cv.imread(img)
    masked = image.copy()
    img_name = os.path.basename(img)
    image_left = int(img_name.split('_')[2])
    image_upper = int(img_name.split('_')[3])
    image_right = int(img_name.split('_')[4])
    image_lower = int(img_name.split('_')[5][:-4])

    labels = os.path.join(label_path, img_name.split('_')[0]+'_ROI_Extract.csv')
    df = pd.read_csv(labels, delimiter='\t')

    counts = [0, 0]
    for index, row in df.iterrows():
        id = row['name']
        type = row['type']
        top = row['top']
        left = row['left']
        bottom = row['bottom']
        right = row['right']
        n_coordinates = row['n_coordinates']
        integer_coordinates = row['integer_coordinates']
        subpixel_coordinates = row['subpixel_coordinates']

        if not contained_within((left, top, right, bottom), (image_left, image_upper, image_right, image_lower)):
            counts[0] += 1
            continue
        counts[1] += 1
        cords = []
        try:
            for i in subpixel_coordinates.split('\n'):
                matches = re.findall(r'[-+]?\d*\.\d+|\d+', i)
                if len(matches) == 2:
                    y = float(matches[1])
                    x = float(matches[0])
                    cords.append((x, y))
        except Exception as e:
            print(f"Error drawing polygons for {img_name}: {e}")
            continue
        
        shifted_box = [left - image_left, top - image_upper, right-image_left, bottom-image_upper]
        cv.rectangle(image, (shifted_box[0], shifted_box[1]), (shifted_box[2], shifted_box[3]), (255, 0, 0), 2)
        cv.putText(image, type, (shifted_box[0], shifted_box[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        #polygons
        desired_color = (0, 255, 0)
        points = np.array(cords, np.int32).reshape(-1,2)
        shifted_points = points - [image_left, image_upper]
        cv.fillPoly(masked, [points], desired_color)
        
    print(f"Total annotations: {counts[0] + counts[1]}, Annotations within image: {counts[1]}")
    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.join(imgsave_path, img_name[:-4] + '.jpg')    
    cv.imwrite(imgsavename, semi_transparent_mask)


    import code
    code.interact(local={**locals(), **globals()})

