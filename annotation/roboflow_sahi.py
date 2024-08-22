#! /usr/bin/env python3

""" robolflow_sahi_test.py
trying to use roboflow to annotate images for sahi segemntation
following this blog https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/ 
"""

##TODO add to yaml setup file
#pip install supervision <- actually needed

import supervision as sv
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
import zipfile
import torch
from Utils import classes, class_colours

weight_file = '/home/java/Java/ultralytics/runs/segment/train10/weights/best.pt' #trained on labeled list iteration 1
weight_file = '/home/java/Java/ultralytics/runs/segment/train12/weights/best.pt' #trained on labeled list iteration 1 with albmumentations
weight_file = '/home/java/Java/ultralytics/runs/segment/train13/weights/best.pt' #trained on labeled list iteration 1 and 2 with albmumentations
#weight_file = '/home/java/Java/ultralytics/runs/segment/train17/weights/best.pt' #trained on labeled list iteration 1 and 2 with albumentations no parital labels
base_img_location = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/cgras_data_copied_2240605_ultralytics_data/images'
save_dir = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/ultralytics_data_detections'
base_file = "/home/java/Downloads/AugCgras/annotations.xml"
output_filename = "/home/java/Downloads/cgras_2Aug20.xml"
#list of images that have already been labeled
labeled_images_iteration1 = [0, 1, 2, 3, 5, 
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 112, 
                             149] + list(range(370, 400)) + list(range(440, 470)) + list(range(550, 580)) +[650, 651, 652, 653, 658] + list(range(700, 724)) + [750, 774]
labeled_images_iteration2 = list(range(6, 99)) + [109, 111, 113, 114, 115]
labeled_images = labeled_images_iteration1 + labeled_images_iteration2
intrested_images = [601, 602, 603, 629, 630, 631, 632, 633, 664, 665, 666, 667, 668, 669] #[506, 507, 508, 509, 510, 528, 529, 530, 531, 532, 581, 582, 583, 584, 585, 601, 602, 603, 606, 607, 608, 609, 610, 611, 629, 630, 631, 632, 633, 664, 665, 666, 667, 668, 669, 727, 728, 729, 730, 731] # range(inclusive, exclusive) #223 missed 
#batch1 = list(range(339, 344)) + list(range(415, 419)) + list(range(474, 480)) + list(range(435, 330))
max_img = 1000
single_image = False #run roboflow sahi on one image and get detected segmentation results
visualise = True #visualise the detections on the images

## FUNCIONS
#quicker then the version in Utils.py #TODO probably update one in utils
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

def callback(image_slice: np.ndarray) -> sv.Detections:
    results = model(image_slice)
    try:
        detections = sv.Detections.from_ultralytics(results[0])
    except:
        print("Error in callback")
        import code
        code.interact(local=dict(globals(), **locals()))
    return detections

def overlap_boxes(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    if x1 > x4 or x3 > x2:
        return False
    if y1 > y4 or y3 > y2:
        return False
    return True

#TODO functionlise / simplify?
def combine_detections(box_array, conf_list, cls_id_list, mask_list):
    updated_box_array, updated_conf_list, updated_class_id, updated_mask_list = [], [], [], []
    overlap_count = 0
    combined_indices = set()
    for i, mask1 in enumerate(mask_list):
        if i in combined_indices:
            continue  # Skip already combined detections
        box1 = box_array[i]
        overlap = False
        for j in range(i + 1, len(mask_list)):
            if j in combined_indices or j<i:
                continue
            mask2 = mask_list[j]
            box2 = box_array[j]
            if overlap_boxes(box1, box2): #assume only pairs overlap
                overlap = True
                overlap_count += 1
                new_box = [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])]
                new_class = cls_id_list[i] if conf_list[i] > conf_list[j] else cls_id_list[j]
                new_conf = (conf_list[i] + conf_list[j]) / 2
                mask1_tl_x, mask1_tl_y, mask1_w, mask1_h = mask1[1], mask1[2], mask1[3], mask1[4]
                mask2_tl_x, mask2_tl_y, mask2_w, mask2_h = mask2[1], mask2[2], mask2[3], mask2[4]
                # New mask
                new_tl_x, new_tl_y = min(mask1_tl_x, mask2_tl_x), min(mask1_tl_y, mask2_tl_y)
                new_w = max(mask1_tl_x + mask1_w, mask2_tl_x + mask2_w) - new_tl_x
                new_h = max(mask1_tl_y + mask1_h, mask2_tl_y + mask2_h) - new_tl_y
                new_mask = np.zeros((new_h, new_w), dtype=np.uint8)
                mask1_x_offset = mask1_tl_x - new_tl_x
                mask1_y_offset = mask1_tl_y - new_tl_y
                new_mask[mask1_y_offset:mask1_y_offset + mask1_h, mask1_x_offset:mask1_x_offset + mask1_w] = mask1[0]
                mask2_x_offset = mask2_tl_x - new_tl_x
                mask2_y_offset = mask2_tl_y - new_tl_y
                new_mask[mask2_y_offset:mask2_y_offset + mask2_h, mask2_x_offset:mask2_x_offset + mask2_w] = np.logical_or(
                    new_mask[mask2_y_offset:mask2_y_offset + mask2_h, mask2_x_offset:mask2_x_offset + mask2_w], mask2[0]
                ).astype(np.uint8)
                print(f"Combining {i} and {j}")
                updated_box_array.append(new_box)
                updated_conf_list.append(new_conf)
                updated_class_id.append(new_class)
                updated_mask_list.append((new_mask, new_tl_x, new_tl_y, new_w, new_h))
                combined_indices.update([i, j])
                break
        if not overlap:
            updated_box_array.append(box1)
            updated_conf_list.append(conf_list[i])
            updated_class_id.append(cls_id_list[i])
            updated_mask_list.append(mask1)
    updated_box_array = np.array(updated_box_array)
    return updated_box_array, updated_conf_list, updated_class_id, updated_mask_list

## OBJECTS
model = YOLO(weight_file)
mask_annotator = sv.MaskAnnotator()
slicer = sv.InferenceSlicer(callback=callback, slice_wh=(640, 640), overlap_ratio_wh=(0.1, 0.1))

##predict and CVAT
class Predict2Cvat:
    BASE_FILE = "/home/java/Java/Cgras/cgras_settler_counter/annotations.xml"
    OUTPUT_FILE = "/home/java/Downloads/complete.xml"
    DEFAULT_WEIGHT_FILE = "/home/java/Java/ultralytics/runs/segment/train9/weights/cgras_yolov8n-seg_640p_20231209.pt"
    DEFAULT_SAVE_DIR = "/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/ultralytics_data_detections"
    DEFAULT_MAX_IMG = 10000
    DEFAULT_BATCH_SIZE = 3000
    
    def __init__(self, 
                 img_location: str, 
                 output_file: str = OUTPUT_FILE, 
                 weights_file: str = DEFAULT_WEIGHT_FILE,
                 base_file: str = BASE_FILE,
                 max_img: int = DEFAULT_MAX_IMG,
                 save_img: bool = False,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 batch_height: int = DEFAULT_BATCH_SIZE,
                 batch_width: int = DEFAULT_BATCH_SIZE,
                 label_img_no: list = None):
        self.img_location = img_location
        self.base_file = base_file
        self.output_file = output_file
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = YOLO(weights_file).to(self.device)
        self.max_img = max_img
        self.save = save_img
        self.save_dir = save_dir
        self.batch_height = batch_height
        self.batch_width = batch_width
        self.label_img_no = label_img_no
        self.batch = True #weather to batch an image or not

    def tree_setup(self):
        tree = ET.parse(self.base_file)
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
        return new_tree, root

    #this is done so that the memory doesn't fill up straight away with the large images
    def batch_image(self, image_cv, image_height, image_width, slicer):
        data_dict = {'class_name': []}
        box_list, conf_list, cls_id_list, mask_list = [], [], [], []
        print("Batching image")
        for y in range(0, image_height, self.batch_height):
            for x in range(0, image_width, self.batch_width):
                y_end = min(y + self.batch_height, image_height)
                x_end = min(x + self.batch_width, image_width)
                img= image_cv[y:y_end, x:x_end]
                sliced_detections = slicer(image=img)
                if sliced_detections.confidence.size == 0:
                    print("No detections found in batch")
                    continue
                for box in sliced_detections.xyxy:
                    box[0] += x
                    box[1] += y
                    box[2] += x
                    box[3] += y
                    box_list.append(box)
                for conf in sliced_detections.confidence:
                    conf_list.append(conf)
                for cls_id in sliced_detections.class_id:
                    cls_id_list.append(cls_id)
                for data in sliced_detections.data['class_name']:
                    data_dict['class_name'].append(data)
                for mask in sliced_detections.mask:
                    mask_resized = cv2.resize(mask.astype(np.uint8), (x_end - x, y_end - y))
                    rows, cols = np.where(mask_resized == 1)
                    if len(rows) > 0 and len(cols) > 0:
                        top_left_y = rows.min()
                        bottom_right_y = rows.max()
                        top_left_x = cols.min()
                        bottom_right_x = cols.max()
                        box_width = bottom_right_x - top_left_x + 1
                        box_height = bottom_right_y - top_left_y + 1
                        sub_mask = mask_resized[top_left_y:bottom_right_y + 1, top_left_x:bottom_right_x + 1]
                        mask_list.append((sub_mask, top_left_x + x, top_left_y + y, box_width, box_height))    
        return box_list, conf_list, cls_id_list, mask_list, data_dict

    def save_img_batch(self, image_cv, box_array, conf_list, cls_id_list, mask_list, image_name):
        height, width, _ = image_cv.shape
        masked = image_cv.copy()
        line_tickness = int(round(width)/600)
        font_size = 2#int(round(line_tickness/2))
        font_thickness = 5#3*(abs(line_tickness-font_size))+font_size
        if conf_list is not None:
            for j, m in enumerate(mask_list):
                contours, _ = cv2.findContours(m[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
                for contour in contours:
                    points = np.squeeze(contour)
                    if len(points.shape) == 1:
                        points = points.reshape(-1, 1, 2)
                    elif len(points.shape) == 2 and points.shape[1] != 2:
                        points = points.reshape(-1, 1, 2)
                    points += np.array([m[1], m[2]]) #shift the points to the correct location
                    cls = classes[int(cls_id_list[j])]
                    desired_color = class_colours[cls]
                    if points is None or not points.any() or len(points) == 0:
                        print(f'mask {j} encountered problem with points {points}, class is {cls}')
                    else: 
                        cv2.fillPoly(masked, [points], desired_color) 
            for t, b in enumerate(box_array):
                cls = classes[int(cls_id_list[t])]
                desired_color = class_colours[cls]
                cv2.rectangle(image_cv, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), tuple(class_colours[cls]), line_tickness)
                cv2.putText(image_cv, f"{conf_list[t]:.2f}: {cls}", (int(b[0]-20), int(b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
        else:
            print(f'No masks found in {image_name}')
        alpha = 0.5
        semi_transparent_mask = cv2.addWeighted(image_cv, 1-alpha, masked, alpha, 0)
        imgsavename = os.path.basename(image_name)
        imgsave_path = os.path.join(self.save_dir, imgsavename[:-4] + '_det_masknbox.jpg')
        cv2.imwrite(imgsave_path, semi_transparent_mask)

    def save_img(self, image, sliced_detections):
        masked = image.copy()
        for detection in sliced_detections:
            xyxy = detection[0].tolist()
            mask_array = detection[1].astype(np.uint8) 
            confidence = detection[2]
            class_id = detection[3]
            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = np.squeeze(contour)
                if len(points.shape) == 1:
                    points = points.reshape(-1, 1, 2)
                elif len(points.shape) == 2 and points.shape[1] != 2:
                    points = points.reshape(-1, 1, 2)
                cls = classes[class_id]
                desired_color = class_colours[cls]
                if points is None or not points.any() or len(points) == 0:
                    print(f'mask encountered problem with points {points}, class is {cls}')
                else: 
                    cv2.fillPoly(masked, [points], desired_color) 
            cv2.putText(image, f"{confidence:.2f}: {cls}", (int(xyxy[0]-20), int(xyxy[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2, desired_color, 5)
        alpha = 0.30
        semi_transparent_mask = cv2.addWeighted(image, 1-alpha, masked, alpha, 0)
        imgsave_path = os.path.join(save_dir, os.path.basename(image_file)[:-4] + '_det.jpg')
        cv2.imwrite(imgsave_path, semi_transparent_mask)

    def display_txt_on_img(self, txt_name):
        img_folder = self.img_location
        imgbasename = os.path.basename(txt_name)[:-17]
        imgname = os.path.join(img_folder, imgbasename + '.jpg')
        image = cv2.imread(imgname)
        height, width, _ = image.shape

        with open(txt_name, "r") as file:
            lines = file.readlines()
            for line in lines:
                data = line.strip().split()
                class_idx = int(data[0])
                conf = float(data[-1])
                points_normalised = [float(val) for val in data[1:-1]]
                try:
                    points = np.array(points_normalised).reshape(-1, 1, 2)
                    points[:, 0, 0] *= width
                    points[:, 0, 1] *= height
                    points = points.astype(int)
                    cv2.polylines(image, [points], True, class_colours[classes[class_idx]], 2)
                    cv2.putText(image, f"{conf:.2f}: {classes[class_idx]}", (points[0, 0, 0]-20, points[0, 0, 1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, class_colours[classes[class_idx]], 2)
                except:
                    print("Error in line with length", len(line))
                    continue
        cv2.imwrite(os.path.join(self.save_dir, imgbasename + '_revis.jpg'), image)

    def save_text(self, image_name, image_width, image_height, conf_list, cls_id_list, mask_list):
        masks = mask_list
        txt_results = []
        for i, r in enumerate(masks):
            txt_result1 = [int(cls_id_list[i])]
            contours, _ = cv2.findContours(r[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            for contour in contours:
                if contour.shape == (1, 1, 2):
                    continue
                points = np.squeeze(contour)
                if len(points.shape) == 1:
                    points = points.reshape(-1, 1, 2)
                elif len(points.shape) == 2 and points.shape[1] != 2:
                    points = points.reshape(-1, 1, 2)
                points += np.array([r[1], r[2]]) #shift the points to the correct location
                #normalise points
                normalized_points = points.astype(float) 
                try:
                    normalized_points[:, 0] /= image_width    
                    normalized_points[:, 1] /= image_height
                    normalized_points = normalized_points.flatten()
                    for point in normalized_points:
                        txt_result1.append(point)
                except:
                    print("Error in normalising points")
                    continue
                txt_result1.append(conf_list[i])
                txt_results.append(txt_result1)
        imgsavename = os.path.basename(image_name)
        imgsave_path = os.path.join(self.save_dir, imgsavename[:-4] + '_det_masknbox.txt')
        with open(imgsave_path, 'w') as file:
            for txt_result in txt_results:
                for item in txt_result:
                    file.write(str(item) + ' ')
                file.write('\n')

    def run(self):
        new_tree, root = self.tree_setup()

        for i, image_element in enumerate(root.findall('.//image')):
            print(i,'images being processed')
            if i>self.max_img:
                print("Hit max img limit")
                break
            img_start_time = time.time()

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

            if self.label_img_no is not None and i not in self.label_img_no: #only label images in the list
                #(self.label_img_no is not None and i in self.label_img_no) or i<113 or i>=114: #don't overwrite already labeled images
                print("skipping image, as already labeled")
                for mask in image_element.findall('.//mask'):
                    mask_elem = SubElement(new_elem, 'mask')
                    mask_elem.set('label', mask.get('label'))
                    mask_elem.set('source', mask.get('source'))
                    mask_elem.set('occluded', mask.get('occluded'))
                    mask_elem.set('rle', mask.get('rle'))
                    mask_elem.set('left', mask.get('left'))
                    mask_elem.set('top', mask.get('top'))
                    mask_elem.set('width', mask.get('width'))
                    mask_elem.set('height', mask.get('height'))
                    mask_elem.set('z_order', mask.get('z_order'))
                continue

            image_file = os.path.join(self.img_location, image_name)
            image_cv = cv2.imread(image_file)
            image_height, image_width = image_cv.shape[:2]
            if self.batch:
                box_list, conf_list, cls_id_list, mask_list, _ = self.batch_image(image_cv, image_height, image_width, slicer)
                conf_array = np.array(conf_list)
                box_array = np.array(box_list)
                box_array, updated_conf_list, cls_id_list, mask_list = combine_detections(box_array, conf_array, cls_id_list, mask_list)
                conf_array = np.array(updated_conf_list)
                box_array, updated_conf_list, cls_id_list, mask_list = combine_detections(box_array, conf_array, cls_id_list, mask_list)
                conf_array = np.array(updated_conf_list)
            else:
                sliced_detections = slicer(image=image_cv)
                conf_array = sliced_detections.confidence

            if self.save:
                if self.batch:
                    self.save_img_batch(image_cv, box_array, conf_list, cls_id_list, mask_list, image_name)
                    self.save_text(image_name, image_width, image_height, conf_list, cls_id_list, mask_list)
                else:
                    self.save_img(image_cv, sliced_detections)
                    
            if conf_array is None:
                print('No masks found in image',image_name)
                continue

            for j, detection in enumerate(conf_array):
                try:
                    if self.batch:
                        sub_mask, top_left_x, top_left_y, box_width, box_height = mask_list[j]
                    else:
                        cls_id_list = sliced_detections.class_id
                        mask = sliced_detections.mask[j].astype(np.uint8)
                        rows, cols = np.where(mask == 1)
                        if len(rows) > 0 and len(cols) > 0:
                            top_left_y = rows.min()
                            bottom_right_y = rows.max()
                            top_left_x = cols.min()
                            bottom_right_x = cols.max()
                            box_width = bottom_right_x - top_left_x + 1
                            box_height = bottom_right_y - top_left_y + 1
                            sub_mask = mask[top_left_y:bottom_right_y + 1, top_left_x:bottom_right_x + 1]
                    rle = binary_mask_to_rle(sub_mask)
                    #rle_to_binary_mask(rle, box_width, box_height, True) #check rle, works
                    rle_string = ', '.join(map(str, rle))
                    label = classes[cls_id_list[j]]#data_dict['class_name'][j]
                    mask_elem = SubElement(new_elem, 'mask')
                    mask_elem.set('label', label)
                    mask_elem.set('source', 'semi-auto')
                    mask_elem.set('occluded', '0')
                    mask_elem.set('rle', rle_string)
                    mask_elem.set('left', str(int(top_left_x)))
                    mask_elem.set('top', str(int(top_left_y)))
                    mask_elem.set('width', str(int(box_width)))
                    mask_elem.set('height', str(int(box_height)))
                    mask_elem.set('z_order', '0')
                except:
                    print(f'detection {j} encountered problem')
                    import code
                    code.interact(local=dict(globals(), **locals()))
            
            new_tree.write(self.output_file, encoding='utf-8', xml_declaration=True) #save as progress incase of crash
            print(len(conf_array),'masks converted in image',image_name, "number", i)
            img_end_time = time.time()
            print('Image run time: {} sec'.format(img_end_time - img_start_time))

        new_tree.write(self.output_file, encoding='utf-8', xml_declaration=True)
        zip_filename = self.output_file.split('.')[0] + '.zip'
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.output_file, arcname='output_xml_file.xml')
        print('XML file zipped')


print("Detecting corals and saving to annotation format")
Det = Predict2Cvat(base_img_location, output_filename, weight_file, base_file, save_img=True, max_img=max_img, label_img_no=intrested_images)
Det.run()
print("Done detecting corals")

import code
code.interact(local=dict(globals(), **locals()))


def save_img(image_cv, box_array, conf_list, cls_id_list, mask_list, image_name, save_dir):
    height, width, _ = image_cv.shape
    masked = image_cv.copy()
    line_tickness = int(round(width)/600)
    font_size = 2#int(round(line_tickness/2))
    font_thickness = 5#3*(abs(line_tickness-font_size))+font_size
    if conf_list is not None:
        for j, m in enumerate(mask_list):
            contours, _ = cv2.findContours(m[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            for contour in contours:
                points = np.squeeze(contour)
                if len(points.shape) == 1:
                    points = points.reshape(-1, 1, 2)
                elif len(points.shape) == 2 and points.shape[1] != 2:
                    points = points.reshape(-1, 1, 2)
                points += np.array([m[1], m[2]]) #shift the points to the correct location
                cls = classes[int(cls_id_list[j])]
                desired_color = class_colours[cls]
                if points is None or not points.any() or len(points) == 0:
                    print(f'mask {j} encountered problem with points {points}, class is {cls}')
                else: 
                    cv2.fillPoly(masked, [points], desired_color) 
        for t, b in enumerate(box_array):
            cls = classes[int(cls_id_list[t])]
            desired_color = class_colours[cls]
            cv2.putText(image_cv, f"{conf_list[t]:.2f}: {cls}", (int(b[0]-20), int(b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    else:
        print(f'No masks found in {image_name}')
    alpha = 0.5
    semi_transparent_mask = cv2.addWeighted(image_cv, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.basename(image_name)
    imgsave_path = os.path.join(save_dir, imgsavename[:-4] + '_det_mask.jpg')
    cv2.imwrite(imgsave_path, semi_transparent_mask)

batch = True
batch_height, batch_width = 3000, 3000

## Visualise Detections on images
if visualise:
    print("visulising detections")
    os.makedirs(save_dir, exist_ok=True)
    imglist = sorted(glob.glob(os.path.join(base_img_location, '*.jpg')))
    for i, image_file in enumerate(imglist):
        print(f"processing image: {i+1} of {len(imglist)}")
        if not i == 43:
            continue
        if i>max_img:
            print("Hit max img limit")
            break
        image = cv2.imread(image_file)
        image_height, image_width = image.shape[:2]
        if batch:
            data_dict = {'class_name': []}
            box_list, conf_list, cls_id_list, mask_list = [], [], [], []
            print("Batching image")
            for y in range(0, image_height, batch_height):
                for x in range(0, image_width, batch_width):
                    y_end = min(y + batch_height, image_height)
                    x_end = min(x + batch_width, image_width)
                    img= image[y:y_end, x:x_end]
                    sliced_detections = slicer(image=img)
                    if sliced_detections.confidence.size == 0:
                        print("No detections found in batch")
                        continue
                    for box in sliced_detections.xyxy:
                        box[0] += x
                        box[1] += y
                        box[2] += x
                        box[3] += y
                        box_list.append(box)
                    for conf in sliced_detections.confidence:
                        conf_list.append(conf)
                    for cls_id in sliced_detections.class_id:
                        cls_id_list.append(cls_id)
                    for data in sliced_detections.data['class_name']:
                        data_dict['class_name'].append(data)
                    for mask in sliced_detections.mask:
                        mask_resized = cv2.resize(mask.astype(np.uint8), (x_end - x, y_end - y))
                        rows, cols = np.where(mask_resized == 1)
                        if len(rows) > 0 and len(cols) > 0:
                            top_left_y = rows.min()
                            bottom_right_y = rows.max()
                            top_left_x = cols.min()
                            bottom_right_x = cols.max()
                            box_width = bottom_right_x - top_left_x + 1
                            box_height = bottom_right_y - top_left_y + 1
                            sub_mask = mask_resized[top_left_y:bottom_right_y + 1, top_left_x:bottom_right_x + 1]
                            mask_list.append((sub_mask, top_left_x + x, top_left_y + y, box_width, box_height))   
            conf_array = np.array(conf_list)
            box_array = np.array(box_list)
            updated_box_array, updated_conf_list, updated_class_id, updated_mask_list = combine_detections(box_array, conf_array, cls_id_list, mask_list)
            import code
            code.interact(local=dict(globals(), **locals()))
            save_img(image, box_array, conf_list, cls_id_list, mask_list, os.path.basename(image_file), save_dir)
            save_img(image, updated_box_array, updated_conf_list, updated_class_id, updated_mask_list, os.path.basename(image_file), save_dir)
            
        else:
            sliced_detections = slicer(image=image)
            masked = image.copy()
            for detection in sliced_detections:
                xyxy = detection[0].tolist()
                mask_array = detection[1].astype(np.uint8) 
                confidence = detection[2]
                class_id = detection[3]
                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    points = np.squeeze(contour)
                    if len(points.shape) == 1:
                        points = points.reshape(-1, 1, 2)
                    elif len(points.shape) == 2 and points.shape[1] != 2:
                        points = points.reshape(-1, 1, 2)
                    cls = classes[class_id]
                    desired_color = class_colours[cls]
                    if cls == 'grazer_snail':
                        import code
                        code.interact(local=dict(globals(), **locals()))
                    if points is None or not points.any() or len(points) == 0:
                        print(f'mask encountered problem with points {points}, class is {cls}')
                    else: 
                        cv2.fillPoly(masked, [points], desired_color) 
                cv2.putText(image, f"{confidence:.2f}: {cls}", (int(xyxy[0]-20), int(xyxy[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2, desired_color, 5)
            alpha = 0.30
            semi_transparent_mask = cv2.addWeighted(image, 1-alpha, masked, alpha, 0)
            imgsave_path = os.path.join(save_dir, os.path.basename(image_file)[:-4] + '_t1.jpg')
            cv2.imwrite(imgsave_path, semi_transparent_mask)
        import code
        code.interact(local=dict(globals(), **locals()))


### Single image check and test
if single_image:
    image_file = "/home/java/Java/data/cgras_20231028/images/2712-4-1-1-0-231220-1249.jpg"
    image = cv2.imread(image_file)

    start_time = time.time()
    sliced_detections = slicer(image=image)
    end_time = time.time()

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=sliced_detections)

    sv.plot_image(annotated_image) #visualise all the detections

    duration = end_time - start_time
    print('slice run time: {} sec'.format(duration))
    print('slice run time: {} min'.format(duration / 60.0))

    # get detections in yolo format
    start_time = time.time()
    for i, detection in enumerate(sliced_detections):
        print(f"detection: {i+1} of {len(sliced_detections)}")
        xyxy = detection[0].tolist()
        mask_array = detection[1] #Bool array of img_h x img_w
        confidence = detection[2]
        class_id = detection[3]
        rle = binary_mask_to_rle(mask_array) 
        left, top, width, height = min(xyxy[0], xyxy[2]), min(xyxy[1], xyxy[3]), abs(xyxy[0] - xyxy[2]), abs(xyxy[1] - xyxy[3])
        label = detection[5]['class_name']
        #rle_to_binary_mask(rle, 5304, 7952, True) #check rle, works
    end_time = time.time()

    duration = end_time - start_time
    print('detction proecss total run time: {} sec'.format(duration))
    print('detction proecss total run time: {} min'.format(duration / 60.0))
    duration = (end_time - start_time)/len(sliced_detections)
    print('detction proecss average run time: {} sec'.format(duration))
    print('detction proecss average run time: {} min'.format(duration / 60.0))

    import code
    code.interact(local=dict(globals(), **locals()))

