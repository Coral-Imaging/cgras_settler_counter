#! /usr/bin/env python3

""" tiling_images.py
script created to tile big images into smaller ones with annotations that can then be trained on via a yolo model
"""

import os
import numpy as np
import cv2 as cv
import glob
from PIL import Image
from shapely.geometry import Polygon, box, MultiPolygon, GeometryCollection
from shapely.validation import explain_validity
from tqdm import tqdm

#from annotation.Utils import classes, class_colours 
classes = ["recruit_live_white", "recruit_cluster_live_white", "recruit_symbiotic", "recruit_cluster_symbiotic", "recruit_partial",
           "recruit_cluster_partial", "recruit_dead", "recruit_cluster_dead", "grazer_snail", "pest_tubeworm", "unknown"] #how its in cvat
# Colours for each class
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


full_res_dir = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/dec_17_split/train/'
save_path = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/dec_17_split_n_tilled/'

TILE_WIDTH= 640
TILE_HEIGHT = 640
TRUNCATE_PERCENT = 0.5
prefix = 'train'

#images in one folder, labels in another. Only want to do images with an ossociated label file
imglist = sorted(glob.glob(os.path.join(full_res_dir, 'images', '*.jpg')))
TILE_OVERLAP = round((TILE_HEIGHT+TILE_WIDTH)/2 * TRUNCATE_PERCENT)
directory_count = 4
file_counter = 0   

def make_sub_dirctory_save(prefix, save_path):
    save_train = os.path.join(save_path, f'{prefix}_{directory_count}')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_train, exist_ok=True)
    save_images = os.path.join(save_train, 'images')
    save_labels = os.path.join(save_train, 'labels')
    os.makedirs(save_images, exist_ok=True)
    os.makedirs(save_labels, exist_ok=True)
    return save_images, save_labels

save_img, save_labels = make_sub_dirctory_save(prefix, save_path)

def is_mostly_contained(polygon, x_start, x_end, y_start, y_end, threshold):
    """Returns true if a Shaply polygon has more then threshold percent in the area of a specified bounding box."""
    polygon_box = box(*polygon.bounds)
    tile_box = box(x_start, y_start, x_end, y_end)
    if not polygon.is_valid:
        explanation = explain_validity(polygon)
        #print(f"Invalid Polygon: {explanation} at {x_start}_{y_start}")
        return False
    if not polygon_box.intersects(tile_box):
        return False
    intersection = polygon.intersection(tile_box)
    return intersection.area > (threshold * polygon.area)

def truncate_polygon(polygon, x_start, x_end, y_start, y_end):
    """Returns a polygon with points constrained to a specified bounding box."""
    tile_box = box(x_start, y_start, x_end, y_end)
    intersection = polygon.intersection(tile_box)
    return intersection

def create_polygon_unnormalised(parts, img_width, img_height):
    """Creates a Polygon from unnormalized part coordinates, as [class_ix, xn, yn ...]"""
    xy_coords = [round(float(p) * img_width) if i % 2 else round(float(p) * img_height) for i, p in enumerate(parts[1:], start=1)]
    polygon_coords = [(xy_coords[i], xy_coords[i + 1]) for i in range(0, len(xy_coords), 2)]
    polygon = Polygon(polygon_coords)
    return polygon

def normalise_polygon(truncated_polygon, class_number, x_start, x_end, y_start, y_end, width, height):
    """Normalize coordinates of a polygon with respect to a specified bounding box."""
    points = []
    if isinstance(truncated_polygon, Polygon):
        x_coords, y_coords = truncated_polygon.exterior.coords.xy
        xy = [class_number]

        for c, d in zip(x_coords, y_coords):
            x_val = 1.0 if c == x_end else (c - x_start) / width
            y_val = 1.0 if d == y_end else (d - y_start) / height
            xy.extend([x_val, y_val])

        points.append(xy)
        
    elif isinstance(truncated_polygon, (MultiPolygon, GeometryCollection)):
        for p in truncated_polygon.geoms:
            points.append(normalise_polygon(p, class_number, x_start, x_end, y_start, y_end, width, height))
    return points

def cut_n_save_img(x_start, x_end, y_start, y_end, np_img, img_save_path):
    """Save a tile section of an image given by a bounding box"""
    cut_tile = np.zeros(shape=(TILE_WIDTH, TILE_HEIGHT, 3), dtype=np.uint8)
    cut_tile[0:TILE_HEIGHT, 0:TILE_WIDTH, :] = np_img[y_start:y_end, x_start:x_end, :]
    cut_tile_img = Image.fromarray(cut_tile)
    cut_tile_img.save(img_save_path)

def cut_annotation(x_start, x_end, y_start, y_end, lines, imgw, imgh):
    """From instance lines in label file, find objects in the bounding box and return the renormalised xy points if there are any"""
    writeline = []
    incomplete_lines = set()  
    for j, line in enumerate(lines):
        parts = line.split()
        class_number = int(parts[0])
        polygon = create_polygon_unnormalised(parts, imgw, imgh)

        if len(parts) < 1 or polygon.is_empty:
            if j not in incomplete_lines: 
                print(f"line {j} is incomplete")
                incomplete_lines.add(j)
                import code
                code.interact(local=dict(globals(), **locals()))    
            continue
        if is_mostly_contained(polygon, x_start, x_end, y_start, y_end, TRUNCATE_PERCENT):
            truncated_polygon = truncate_polygon(polygon, x_start, x_end, y_start, y_end)
            xyn = normalise_polygon(truncated_polygon, class_number, x_start, x_end, y_start, y_end, TILE_WIDTH, TILE_HEIGHT)
            writeline.append(xyn)
    return writeline


#cut the image
def cut(img_name, save_img, test_name, save_labels, txt_name, img_no):
    """Cut a image into tiles, save the annotations renormalised"""
    pil_img = Image.open(img_name, mode='r')
    np_img = np.array(pil_img, dtype=np.uint8)
    img = cv.imread(img_name)
    imgw, imgh = img.shape[1], img.shape[0]
    x_tiles = (imgw + TILE_WIDTH - TILE_OVERLAP - 1) // (TILE_WIDTH - TILE_OVERLAP)
    y_tiles = (imgh + TILE_HEIGHT - TILE_OVERLAP - 1) // (TILE_HEIGHT - TILE_OVERLAP)
    total_tiles = x_tiles * y_tiles  # Total tiles to process
    with tqdm(total=total_tiles, desc=f"Processing image {os.path.basename(img_name)[:-4]}", unit="tile") as pbar:
        for x in range(x_tiles):
            for y in range(y_tiles):
                x_end = min((x + 1) * TILE_WIDTH - TILE_OVERLAP * (x != 0), imgw)
                x_start = x_end - TILE_WIDTH
                y_end = min((y + 1) * TILE_HEIGHT - TILE_OVERLAP * (y != 0), imgh)
                y_start = y_end - TILE_HEIGHT

                img_save_path = os.path.join(save_img, f"{test_name}_{str(x_start).zfill(4)}_{str(y_start).zfill(4)}.jpg")
                txt_save_path = os.path.join(save_labels, f"{test_name}_{str(x_start).zfill(4)}_{str(y_start).zfill(4)}.txt")
                #make cut and save image
                cut_n_save_img(x_start, x_end, y_start, y_end, np_img, img_save_path)
                #cut annotaion and save
                with open(txt_name, 'r') as file:
                    lines = file.readlines()
                try:
                    writeline = cut_annotation(x_start, x_end, y_start, y_end, lines, imgw, imgh)
                except:
                    print("error in cut_annotations")
                    import code
                    code.interact(local=dict(globals(), **locals()))    

                with open(txt_save_path, 'w') as file:
                    for line in writeline:
                        file.write(" ".join(map(str, line)).replace('[', '').replace(']', '').replace(',', '') + "\n")
                pbar.update(1)
                # import code
                # code.interact(local=dict(globals(), **locals()))        

def calculate_img_section_no(img_name):
    """return the number of tiles made from one image depending on tile width and img size"""
    pil_img = Image.open(img_name, mode='r')
    np_img = np.array(pil_img, dtype=np.uint8)
    img = cv.imread(img_name)
    imgw, imgh = img.shape[1], img.shape[0]
    # Count number of sections to make
    x_tiles = (imgw + TILE_WIDTH - TILE_OVERLAP - 1) // (TILE_WIDTH - TILE_OVERLAP)
    y_tiles = (imgh + TILE_HEIGHT - TILE_OVERLAP - 1) // (TILE_HEIGHT - TILE_OVERLAP)
    return x_tiles*y_tiles


def visualise(imgdir, save_path):
    """Show all the annotations on to a set of cut images and save at save_path"""
    imglist = glob.glob(os.path.join(imgdir, '*.jpg'))
    for i, imgname in enumerate(imglist):
        print(f'visulasing image {i+1}/{len(imglist)}')
        base_name = os.path.basename(imgname[:-4])
        img_name = os.path.join(save_img, base_name+'.jpg')
        txt_name = os.path.join(save_labels, base_name+'.txt')

        image = cv.imread(img_name)
        height, width, _ = image.shape
        #same code as annotation/view_predictions.py in the save_image_predictions_mask function, if groundtruth:
        points_normalised, points, class_idx = [], [], []
        with open(txt_name, "r") as file:
            lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            class_idx.append(int(data[0]))
            points_normalised.append([float(val) for val in data[1:]])
        for data in points_normalised:
            values = []
            try:
                for i in range(0, len(data), 2):
                    x = round(data[i]*width)
                    y = round(data[i+1]*height)
                    values.extend([x,y])
                points.append(values)
            except:
                points.append(values)
                print(f'invalid line there is {len(data)} data, related to img {base_name}')
                # import code
                # code.interact(local=dict(globals(), **locals())) 
        for idx in range(len(class_idx)):
            pointers = np.array(points[idx], np.int32).reshape(-1,2)
            cv.polylines(image, [pointers], True, class_colours[classes[class_idx[idx]]], 2)
            cv.putText(image, classes[class_idx[idx]], (pointers[0][0], pointers[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, class_colours[classes[class_idx[idx]]], 2)
        imgsavename = os.path.basename(img_name)
        imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det.jpg')
        cv.imwrite(imgsave_path, image)
        # import code
        # code.interact(local=dict(globals(), **locals()))

max_files = 16382
for i, img in enumerate(imglist):
    if i < 0:
        continue
    name = os.path.basename(img)[:-4]
    img_name = os.path.join(full_res_dir,'images', name+'.jpg')
    txt_name = os.path.join(full_res_dir,'labels', name+'.txt')

    files_to_add = calculate_img_section_no(img_name)
    if file_counter+files_to_add >= max_files:
        print(f'Directory full, as {file_counter} files already made and {files_to_add} will be made with next image')
        directory_count += 1
        file_counter = 0
        save_img, save_labels = make_sub_dirctory_save(prefix, save_path)

    print(f"Looking for {txt_name}")
    if os.path.exists(txt_name):
        print(f'cutting image {i+1}/{len(imglist)}')
        cut(img_name, save_img, name, save_labels, txt_name, i)
        file_counter += files_to_add
    else:
        print("no text file for image")
print("done")
import code
code.interact(local=dict(globals(), **locals())) 
# vis_save_path = os.path.join(save_path, 'vis')
# visualise(save_img, vis_save_path)


# ######### With one image #############
# test_name = '00_20230116_MIS1_RC_Aspat_T04_08'
# img_name = os.path.join(full_res_dir,'images', test_name+'.jpg')
# txt_name = os.path.join(full_res_dir,'labels', test_name+'.txt')
# cut(img_name, save_img, test_name, save_labels)
# print("done cutting test image")

# visualise(save_img)
# print("done visualise")
# import code
# code.interact(local=dict(globals(), **locals())) 