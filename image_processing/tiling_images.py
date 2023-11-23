#! /usr/bin/env python3

""" tiling_images.py
script created to til big images into smaller ones with annotations that can then be trained on via a yolo model
"""

import os
import numpy as np
import cv2 as cv
import glob
from PIL import Image
from shapely.geometry import Polygon, box, MultiPolygon, GeometryCollection
from shapely.ops import unary_union



TILE_WIDTH= 640
TILE_HEIGHT = 640
TILE_OVERLAP = 0
TRUNCATE_PERCENT = 0.01

full_res_dir = '/home/java/Java/data/cgras_20230421/train'
save_path = '/home/java/Java/data/cgras_20230421/tilling'
save_train = os.path.join(save_path, 'train')
save_img = os.path.join(save_train, 'images')
save_labels = os.path.join(save_train, 'labels')
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_train, exist_ok=True)
os.makedirs(save_img, exist_ok=True)
os.makedirs(save_labels, exist_ok=True)
test_name = '00_20230116_MIS1_RC_Aspat_T04_08'

img_name = os.path.join(full_res_dir,'images', test_name+'.jpg')
txt_name = os.path.join(full_res_dir,'labels', test_name+'.txt')

#open image
pil_img = Image.open(img_name, mode='r')
np_img = np.array(pil_img, dtype=np.uint8)
img = cv.imread(img_name)
imgw, imgh = img.shape[1], img.shape[0]
# Count number of sections to make
x_tiles = (imgw + TILE_WIDTH - TILE_OVERLAP - 1) // (TILE_WIDTH - TILE_OVERLAP)
y_tiles = (imgh + TILE_HEIGHT - TILE_OVERLAP - 1) // (TILE_HEIGHT - TILE_OVERLAP)

#is polgon in tile
def is_mostly_contained(polygon, x_start, x_end, y_start, y_end, threshold):
    polygon_box = box(*polygon.bounds)
    tile_box = box(x_start, y_start, x_end, y_end)
    if not polygon_box.intersects(tile_box):
        return False
    intersection = polygon.intersection(tile_box)
    return intersection.area > (threshold * polygon.area)

def truncate_polygon(polygon, x_start, x_end, y_start, y_end):
    tile_box = box(x_start, y_start, x_end, y_end)
    intersection = polygon.intersection(tile_box)
    return intersection

def create_polygon_unnormalised(parts):
    xy1 = []
    for i, p in enumerate(parts[1:], start=1):
        if i % 2:
            x1 = round(float(p) * imgw)
            xy1.append(x1)
        else:
            y1 = round(float(p) * imgh)
            xy1.append(y1)
    polygon = Polygon([(xy1[i], xy1[i + 1]) for i in range(0, len(xy1), 2)])
    return polygon

def normalise_polygon(truncated_polygon, class_number, x_start, x_end, y_start, y_end, width, height):
    points = []
    if isinstance(truncated_polygon, Polygon):
        x_coords, y_coords = truncated_polygon.exterior.coords.xy
        x2, y2, xy2 = [], [], [class_number]

        for c in x_coords:
            if c==x_end:
                x2.append(1)
            else:
                x2.append((c%width)/width)
        for d in y_coords:
            if d==y_end:
                y2.append(1)
            else:
                y2.append((d%height)/height)

        for i in range(0,len(x2)):
            xy2.append(x2[i])
            xy2.append(y2[i])
        points.append(xy2)
    elif isinstance(truncated_polygon, (MultiPolygon, GeometryCollection)):
        for p in truncated_polygon.geoms:
            points.append(normalise_polygon(p, class_number, x_start, x_end, y_start, y_end, width, height))
    return points


#cut the image
def cut(x_tiles, y_tiles, save_img, test_name, save_labels):
    for x in range(x_tiles):
        for y in range(y_tiles):
            x_end = min((x + 1) * TILE_WIDTH - TILE_OVERLAP * (x != 0), imgw)
            x_start = x_end - TILE_WIDTH
            y_end = min((y + 1) * TILE_HEIGHT - TILE_OVERLAP * (y != 0), imgh)
            y_start = y_end - TILE_HEIGHT

            img_save_path = os.path.join(save_img,test_name+'_'+str(y_start)+'_'+str(x_start)+'.jpg')
            txt_save_path = os.path.join(save_labels, test_name+'_'+str(y_start)+'_'+str(x_start)+'.txt')

            #make cut and save image
            cut_tile = np.zeros(shape=(TILE_WIDTH, TILE_HEIGHT, 3), dtype=np.uint8)
            cut_tile[0:TILE_HEIGHT, 0:TILE_WIDTH, :] = np_img[y_start:y_end, x_start:x_end, :]
            cut_tile_img = Image.fromarray(cut_tile)
            cut_tile_img.save(img_save_path)

            #cut annotaion and save
            writeline = []
            with open(txt_name, 'r') as file:
                lines = file.readlines()
            for line in lines:
                parts = line.split()
                class_number = int(parts[0])
                polygon = create_polygon_unnormalised(parts)

                if is_mostly_contained(polygon, x_start, x_end, y_start, y_end, TRUNCATE_PERCENT):
                    truncated_polygon = truncate_polygon(polygon, x_start, x_end, y_start, y_end)
                    xyn = normalise_polygon(truncated_polygon, class_number, x_start, x_end, y_start, y_end, TILE_WIDTH, TILE_HEIGHT)
                    writeline.append(xyn)
                    if x_start == 4480 and y_start == 4272: 
                        import code
                        code.interact(local=dict(globals(), **locals()))

            with open(txt_save_path, 'w') as file:
                for line in writeline:
                    file.write(" ".join(map(str, line)).replace('[', '').replace(']', '').replace(',', '') + "\n")
            
            # import code
            # code.interact(local=dict(globals(), **locals()))        

cut(x_tiles, y_tiles, save_img, test_name, save_labels)
print("done cutting test image")
import code
code.interact(local=dict(globals(), **locals())) 

#visualise the result
imglist = glob.glob(os.path.join(save_img, '*.jpg'))
for i, imgname in enumerate(imglist):
    base_name = os.path.basename(imgname[:-4])
    img_name = os.path.join(save_img, base_name+'.jpg')
    txt_name = os.path.join(save_labels, base_name+'.txt')

    image = cv.imread(img_name)
    height, width, _ = image.shape
    points_normalised, points, class_idx = [], [], []

    with open(txt_name, "r") as file:
        lines = file.readlines()
    for line in lines:
        data = line.strip(' \n').split(' ')
        class_idx.append(int(data[0]))
        points_normalised.append([float(val) for val in data[1:]])
    for data in points_normalised:
        values = []
        for i in range(0, len(data), 2):
            x = round(data[i]*width)
            y = round(data[i+1]*height)
            values.extend([x,y])
        points.append(values)
    for idx in range(len(class_idx)):
        pointers = np.array(points[idx], np.int32).reshape(-1,2)
        cv.polylines(image, [pointers], True, [255, 255, 255], 2)
    imgsavename = os.path.basename(img_name)
    imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det.jpg')
    cv.imwrite(imgsave_path, image)

    # import code
    # code.interact(local=dict(globals(), **locals()))

print("done visualise")
