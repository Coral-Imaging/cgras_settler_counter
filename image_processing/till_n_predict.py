#! /usr/bin/env python3

""" tiling_images.py
script created to tile big images into smaller ones and then show the model prediction, assumed model trained on 640p images
"""

import os
import numpy as np
import cv2 as cv
import glob
import torch
from PIL import Image
from shapely.geometry import Polygon, box, MultiPolygon, GeometryCollection
from shapely.validation import explain_validity
from shapely.affinity import translate
from ultralytics import YOLO


classes = ["recruit_live_white", "recruit_cluster_live_white", "recruit_symbiotic", "recruit_symbiotic_cluster", "recruit_partial",
           "recruit_cluster_partial", "recruit_dead", "recruit_cluster_dead", "grazer_snail", "pest_tubeworm", "unknown", "combined"]
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
                classes[10]: dark_green,
                classes[11]: brown}

TILE_WIDTH= 640
TILE_HEIGHT = 640
TRUNCATE_PERCENT = 0.5
TILE_OVERLAP = round((TILE_HEIGHT+TILE_WIDTH)/2 * TRUNCATE_PERCENT)
OBJ_DISTANCE = 10

#full_res_dir = '/home/java/Java/data/cgras_20231028'
full_res_dir = '/home/java/Java/data/cgras_20230421/train'
#save_path = '/home/java/Java/data/cgras_20231028/tilling'
save_path = '/home/java/Java/data/cgras_20230421/train'
weights_file_path = '/home/java/Java/ultralytics/runs/segment/train6/weights/best.pt' #trained on tilled images
save_det = os.path.join(save_path, 'detections')
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_det, exist_ok=True)

def create_polys(results, image, conf, class_list, x_shift, y_shift):
    """Creates a polygon from the predicted results shifted correctly by tile amount as well as storing the polygons class and confidence"""
    height, width, _ = image.shape
    polygons = []
    if results and results[0].masks:
        for j, m in enumerate(results[0].masks):
            xyn = np.array(m.xyn)
            xyn[0, :, 0] = (xyn[0, :, 0] * width)
            xyn[0, :, 1] = (xyn[0, :, 1] * height)
            points = xyn.reshape((-1, 2))
            polygon = Polygon(points)
            translated_polygon = translate(polygon, x_shift, y_shift)
            polygons.append([translated_polygon, conf[j], class_list[j]])
    return polygons                                

def find_objects(img_name, obj_list):
    """From an image returns a list of polygons represented detected corals in tile trunks"""
    pil_img = Image.open(img_name, mode='r')
    np_img = np.array(pil_img, dtype=np.uint8)
    img = cv.imread(img_name)
    imgw, imgh = img.shape[1], img.shape[0]
    # Count number of sections to make
    x_tiles = (imgw + TILE_WIDTH - TILE_OVERLAP - 1) // (TILE_WIDTH - TILE_OVERLAP)
    y_tiles = (imgh + TILE_HEIGHT - TILE_OVERLAP - 1) // (TILE_HEIGHT - TILE_OVERLAP)
    for x in range(x_tiles):
        for y in range(y_tiles):
            x_end = min((x + 1) * TILE_WIDTH - TILE_OVERLAP * (x != 0), imgw)
            x_start = x_end - TILE_WIDTH
            y_end = min((y + 1) * TILE_HEIGHT - TILE_OVERLAP * (y != 0), imgh)
            y_start = y_end - TILE_HEIGHT

            #make cut 
            cut_tile = np.zeros(shape=(TILE_WIDTH, TILE_HEIGHT, 3), dtype=np.uint8)
            cut_tile[0:TILE_HEIGHT, 0:TILE_WIDTH, :] = np_img[y_start:y_end, x_start:x_end, :]

            #get predictions
            results = model.predict(source=cut_tile, iou=0.5, agnostic_nms=True, imgsz=640)
            conf, class_list = [], [] 
            for j, b in enumerate(results[0].boxes):
                conf.append(b.conf.item())
                class_list.append(b.cls.item())
            
            #get polygon of predictions
            polygons = create_polys(results, cut_tile, conf, class_list, x_start, y_start)
            if not len(conf)==0 and not len(class_list)==0:
                obj_list.append(polygons)  
    return obj_list  

def split_detection_object(obj_list):
    """Takes discovered objects, splits the classification and confidence from the polygon"""
    polygons, conf, cls = [], [], []
    for objs in obj_list:
        if len(objs) == 1:
            polygons.append(objs[0][0])
            conf.append(objs[0][1])
            cls.append(objs[0][2])
        else:
            for obj in objs:
                polygons.append(obj[0])
                conf.append(obj[1])
                cls.append(obj[2])
    return polygons, conf, cls

def handle_detection_object(obj_list):
    #merges polygons that are close together and returns all detections in a format that can be visualised
    #TODO work out if better way to merge conf and cls. (currently picking the section with the highest conf)??
    all_polygons = []
    polygons, conf, cls = split_detection_object(obj_list)

    while polygons:
        current_polygon = polygons.pop(0)
        current_conf = conf.pop(0)
        current_cls = cls.pop(0)
        neighbors = [
            (poly, neighbor_conf, neighbor_cls)
            for poly, neighbor_conf, neighbor_cls in zip(polygons, conf, cls)
            if current_polygon.distance(poly) <= OBJ_DISTANCE
        ]
        if neighbors:
            for neighbor, neighbor_conf, neighbor_cls in neighbors:
                current_polygon = current_polygon.union(neighbor).convex_hull
                polygon_index = polygons.index(neighbor)
                polygons.pop(polygon_index)
                conf.pop(polygon_index)
                cls.pop(polygon_index)
                print(f'neighbor conf {neighbor_conf:.2f}, current conf {current_conf:.2f}')
                print(f'neighbor class {classes[int(neighbor_cls)]}, current class {classes[int(current_cls)]}')
                if neighbor_conf > current_conf:
                    current_conf = neighbor_conf
                    current_cls = neighbor_cls
                #current_cls = 11 #to visualise what objects are combined
            all_polygons.append((current_polygon, current_conf, current_cls))
        else:
            all_polygons.append((current_polygon, current_conf, current_cls))
    polygons, conf, cls = zip(*all_polygons)
    return polygons, conf, cls

def stich(obj_list, img_name, save_path):
    """Displays the detected objects onto the image"""
    image = cv.imread(img_name)
    height, width, _ = image.shape
    masked = image.copy()
    line_tickness = int(round(width)/600)
    font_size = int(round(line_tickness/10))
    font_thickness = int((abs(line_tickness-font_size))/3)

    polygons, conf, cls = handle_detection_object(obj_list)

    #draw / visualise the polygons
    for i, polygon in enumerate(polygons):
        xy = polygon.exterior.coords.xy
        xy_int = np.array(xy, np.int32)
        current_cls = classes[int(cls[i])]
        desired_color = class_colours[current_cls]
        polygon_pts = np.column_stack((xy_int[0], xy_int[1])).reshape((-1, 1, 2))
        cv.fillPoly(masked, [polygon_pts], desired_color)
        xmin = min(xy[0])
        ymin = min(xy[1])
        cv.putText(image, f"{conf[i]:.2f}: {current_cls}", (int(xmin-20), int(ymin - 5)), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)

    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.basename(img_name)
    imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det_merged.jpg')
    cv.imwrite(imgsave_path, semi_transparent_mask)
    print(f'saved img at {imgsave_path}')

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_file_path).to(device)

######### With one image #############
#test_name = '00_20230116_MIS1_RC_Aspat_T04_08'
test_name = '03_775_20211213_108'
img_name = os.path.join(full_res_dir,'images', test_name+'.jpg')
obj_list = []
obj_list = find_objects(img_name, obj_list)
print("done cutting test image")
stich(obj_list, img_name, save_det)




# visualise(save_img)
# print("done visualise")
# import code
# code.interact(local=dict(globals(), **locals())) 

# imglist = sorted(glob.glob(os.path.join(full_res_dir, 'images', '*.jpg')))
# for i, img in enumerate(imglist):
#     print(f'cutting image {i+1}/{len(imglist)}')
#     # if i > 20:
#     #     break
#     #     import code
#     #     code.interact(local=dict(globals(), **locals())) 
#     name = os.path.basename(img)[:-4]
#     img_name = os.path.join(full_res_dir,'images', name+'.jpg')
#     txt_name = os.path.join(full_res_dir,'labels', name+'.txt')
#     cut(img_name, save_img, name, save_labels, txt_name)

# visualise(save_img)
import code
code.interact(local=dict(globals(), **locals())) 