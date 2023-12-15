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
from shapely.geometry import Polygon
from shapely.affinity import translate
from ultralytics import YOLO
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
import zipfile
import matplotlib.pyplot as plt
import time

##TODO: import from Utils in annnotations?
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

######### Constants and directories #############
TILE_WIDTH= 640
TILE_HEIGHT = 640
TRUNCATE_PERCENT = 0.5
TILE_OVERLAP = round((TILE_HEIGHT+TILE_WIDTH)/2 * TRUNCATE_PERCENT)
OBJ_DISTANCE = 10

#large_images_dir = '/home/java/Java/data/cgras_20231028'
large_images_dir = '/home/java/Java/data/cgras_20230421/train'
#save_path = '/home/java/Java/data/cgras_20231028/tilling'
save_path = '/home/java/Java/data/cgras_20230421/train'
weights_file_path = '/home/java/Java/ultralytics/runs/segment/train6/weights/best.pt' #trained on tilled images
save_det = os.path.join(save_path, 'detections')
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_det, exist_ok=True)
# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_file_path).to(device)


################## functions ##########################
def create_polys(results, image, conf, class_list, x_shift, y_shift):
    """Creates a polygon from the predicted results shifted correctly by tile amount as well as storing the polygons class and confidence.
       Assumes results are in yolo mask format and the conf and cls are lists in the same order as the mask detections
    """
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

def find_objects(full_image_path):
    """From an image file path returns a list of polygons representing detected corals in tile trunks
       Will cut the image into tiles and then run the model on each tile using yolo.predict
       """
    pil_img = Image.open(full_image_path, mode='r')
    np_img = np.array(pil_img, dtype=np.uint8)
    img = cv.imread(full_image_path)
    imgw, imgh = img.shape[1], img.shape[0]
    obj_list = []
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
    """Merges polygons in the object_list that are close together and returns all detections in a format that can be visualised"""
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
                # print(f'neighbor conf {neighbor_conf:.2f}, current conf {current_conf:.2f}')
                # print(f'neighbor class {classes[int(neighbor_cls)]}, current class {classes[int(current_cls)]}')
                if neighbor_conf > current_conf:
                    current_conf = neighbor_conf
                    current_cls = neighbor_cls
                #current_cls = 11 #to visualise what objects are combined
            all_polygons.append((current_polygon, current_conf, current_cls))
        else:
            all_polygons.append((current_polygon, current_conf, current_cls))
    if not all_polygons or len(all_polygons)==0:
        return [], [], []
    polygons, conf, cls = zip(*all_polygons)
    return polygons, conf, cls

def stich_n_vis(obj_list, img_name, save_path):
    """Takes the detected corals over the cut images and stiches them back together, visualising it on the starting image"""
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



######### With one image #############
# #test_name = '00_20230116_MIS1_RC_Aspat_T04_08'
# test_name = '03_775_20211213_108'
# img_name = os.path.join(large_images_dir,'images', test_name+'.jpg')
# obj_list = find_objects(img_name)
# print("done cutting test image")
# stich(obj_list, img_name, save_det)
# import code
# code.interact(local=dict(globals(), **locals())) 

############# With folder of images ###########
# works, 
# NOTE: is it really better then just running the model on the full image, marks are more detailed and accurate, but more corals are missed??
# image_file_list = sorted(glob.glob(os.path.join(large_images_dir,'images','*.jpg')))

# for i, image_file in enumerate(image_file_list):
#     if i>10:
#         break
    
#     obj_list = find_objects(image_file)
#     print(f'done cutting on {os.path.basename(image_file)}')
#     stich_n_vis(obj_list, image_file, save_det)
#     print(f'done stiching on {os.path.basename(image_file)}')

############## Pretend human in the loop ##############
#works
start_time = time.time()
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

#folder of new images
large_images_dir = '/home/java/Java/data/cgras_20231028'
img_location = os.path.join(large_images_dir,'images')
#images were in cvat and then downloaded in CVAT format
base_ann_file = "/home/java/Downloads/cgras_20231028_no_ann/annotations.xml"
output_file = "/home/java/Downloads/cgras_20231028.xml"
SAVE_AS_MASK = False

tree = ET.parse(base_ann_file)
root = tree.getroot() 
new_tree = ElementTree(Element("annotations"))
# add version element
version_element = ET.Element('version')
version_element.text = '1.1'
new_tree.getroot().append(version_element)
# add Meta elements, (copy over from source_file)
meta_element = root.find('.//meta')
if meta_element is not None:
    new_meta_elem = ET.SubElement(new_tree.getroot(), 'meta')
    # copy all subelements of meta
for sub_element in meta_element:
    new_meta_elem.append(sub_element)
    for i, image_element in enumerate(root.findall('.//image')):
        print(i,'images being processed')
        image_id = image_element.get('id')
        image_name = image_element.get('name')
        image_width = int(image_element.get('width'))
        image_height = int(image_element.get('height'))

        # create new image element in new XML
        new_elem = SubElement(new_tree.getroot(), 'image')
        new_elem.set('id', image_id)
        new_elem.set('name', image_name)
        new_elem.set('width', str(image_width))
        new_elem.set('height', str(image_height))

        image_file = os.path.join(img_location, image_name)
        obj_list = find_objects(image_file)
        print(f'done cutting on {os.path.basename(image_file)}')
        polygons, conf, cls = handle_detection_object(obj_list)
        
        for j, polygon in enumerate(polygons):
            label = classes[int(cls[j])]
            xy = polygon.exterior.coords.xy
            xy_float = np.array(xy, np.float32)
            polygon_pts = xy_float.T.reshape((-1, 2))
            if SAVE_AS_MASK:
                try:
                    rle_string, left, top, width, height  = poly_2_rle(polygon_pts,", ",False)
                    mask_elem = SubElement(new_elem, 'mask')
                    mask_elem.set('label', label)
                    mask_elem.set('source', 'semi-auto')
                    mask_elem.set('occluded', '0')
                    mask_elem.set('rle', rle_string)
                    mask_elem.set('left', str(left))
                    mask_elem.set('top', str(top))
                    mask_elem.set('width', str(width))
                    mask_elem.set('height', str(height))
                    mask_elem.set('z_order', '0')
                except:
                    print("error encountered converting polygon to rle")
                    print(f'polygon = {polygon}')
                    import code
                    code.interact(local=dict(globals(), **locals()))
            else:
                if polygon_pts is None or len(polygon_pts)==0:
                    print("error encountered converting polygon to rle")
                    print(f'polygon = {polygon}')
                    import code
                    code.interact(local=dict(globals(), **locals()))
                else:
                    formatted_points = ';'.join([f"{x:.2f},{y:.2f}" for x, y in polygon_pts if x and y])
                    mask_elem = SubElement(new_elem, 'polygon')
                    mask_elem.set('label', label)
                    mask_elem.set('source', 'manual')
                    mask_elem.set('occluded', '0')
                    mask_elem.set('points', formatted_points)
                    mask_elem.set('z_order', '0')
        
        print(f'{len(conf)} objects converted in image {os.path.basename(image_file)}')

    new_tree.write(output_file, encoding='utf-8', xml_declaration=True)

    zip_filename = output_file.split('.')[0] + '.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname='output_xml_file.xml')
    print('XML file zipped')

    end_time = time.time()
    print(f'time taken = {end_time-start_time:.2f} seconds')
    import code
    code.interact(local=dict(globals(), **locals()))