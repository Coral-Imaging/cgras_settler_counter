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

weight_file = '/home/java/Java/ultralytics/runs/segment/train10/weights/best.pt'
base_img_location = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/cgras_data_copied_2240605_ultralytics_data/images'
save_dir = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/ultralytics_data_detections'
base_file = "/home/java/Downloads/cgras20240716/annotations.xml"
output_filename = "/home/java/Downloads/cgras_2024_1.xml"
max_img = 10000
single_image = False #run roboflow sahi on one image and get detected segmentation results
visualise = True #visualise the detections on the images
batch_height, batch_width = 3000, 3000

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
    
    def __init__(self, 
                 img_location: str, 
                 output_file: str = OUTPUT_FILE, 
                 weights_file: str = DEFAULT_WEIGHT_FILE,
                 base_file: str = BASE_FILE,
                 max_img: int = DEFAULT_MAX_IMG,
                 save_img: bool = False,
                 save_dir: str = DEFAULT_SAVE_DIR):
        self.img_location = img_location
        self.base_file = base_file
        self.output_file = output_file
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = YOLO(weights_file).to(self.device)
        self.max_img = max_img
        self.save_img = save_img
        self.save_dir = save_dir

    def wip_vis(self, image_cv, binary_mask_resized, image_file):
        overlay = np.zeros_like(image_cv, dtype=np.uint8)
        overlay[binary_mask_resized == 1] = [0, 0, 255]
        cv2.imwrite(f"{os.path.join(self.save_dir,os.path.basename(image_file)[:-4])}_det.jpg",overlay)

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
    def batch_image(self, image_cv, batch_height, batch_width, image_height, image_width, slicer):
        data_dict = {'class_name': []}
        box_list, conf_list, cls_id_list, mask_list = [], [], [], []
        whole_image_mask = np.zeros((image_height, image_width), dtype=bool)
        print("Batching image")
        for y in range(0, image_height, batch_height):
            for x in range(0, image_width, batch_width):
                y_end = min(y + batch_height, image_height)
                x_end = min(x + batch_width, image_width)
                img= image_cv[y:y_end, x:x_end]
                sliced_detections = slicer(image=img)
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
                    mask_resized = cv2.resize(mask.astype(np.uint8), (batch_width, batch_height))
                    mask_list.append(mask_resized)
                whole_image_mask[y:y_end, x:x_end] = 0
        return box_list, conf_list, cls_id_list, mask_list, data_dict, sliced_detections

    def run(self):
        new_tree, root = self.tree_setup()

        for i, image_element in enumerate(root.findall('.//image')):
            print(i+1,'images being processed')
            if i>self.max_img:
                print("Hit max img limit")
                break
            if i<6 or i>7:
                print("skipping image, as already labeled")
                continue
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
            
            image_file = os.path.join(self.img_location, image_name)
            image_cv = cv2.imread(image_file)
            image_height, image_width = image_cv.shape[:2]
            box_list, conf_list, cls_id_list, mask_list, data_dict, sliced_detections = self.batch_image(image_cv, batch_height, batch_width, image_height, image_width, slicer)
            box_array = np.array(box_list)
            conf_array = np.array(conf_list)
            mask_array = np.array(mask_list)

            if self.save_img:
                print("stiching batch back together")
                cls_id_array = np.array(cls_id_list)
                whole_image_detection = sv.Detections(xyxy=box_array, confidence=conf_array, class_id=cls_id_array, 
                                                    mask=mask_array, data=data_dict)
                annotated_image = mask_annotator.annotate(scene=image.copy(), detections=whole_image_detection)
                cv2.imwrite(f"{os.path.join(self.save_dir,os.path.basename(image_file)[:-4])}_det.jpg", annotated_image)

            if conf_array is None:
                print('No masks found in image',image_name)
                continue

            for j, detection in enumerate(box_array):
                try:
                    xyxy = detection.tolist()
                    binary_mask = mask_array[j]
                    binary_mask_resized = cv2.resize(binary_mask, (image_cv.shape[1], image_cv.shape[0]))
                    rle = binary_mask_to_rle(binary_mask_resized) #<- NOTE error here probably?
                    #rle starts with too big a number. Which I think is the source of the problem, but not sure why this error occurs or how to fix it or where specifically it comes from

                    # #Masks look reasonable from here
                    # self.wip_vis(image_cv, binary_mask, image_file) #test binary masks detected
                    # rle_to_binary_mask(rle, image_width, image_height, True)
                    rle_string = ', '.join(map(str, rle))
                    left, top, width, height = min(xyxy[0], xyxy[2]), min(xyxy[1], xyxy[3]), abs(xyxy[0] - xyxy[2]), abs(xyxy[1] - xyxy[3])
                    label = data_dict['class_name'][j]
                    mask_elem = SubElement(new_elem, 'mask')
                    mask_elem.set('label', label)
                    mask_elem.set('source', 'semi-auto')
                    mask_elem.set('occluded', '0')
                    mask_elem.set('rle', rle_string)
                    # #setting below to 0,0,image_width,image_height causes cvat to crash when trying to view the images
                    mask_elem.set('left', str(int(left)))
                    mask_elem.set('top', str(int(top)))
                    mask_elem.set('width', str(int(width)))
                    mask_elem.set('height', str(int(height)))
                    mask_elem.set('z_order', '0')
                except:
                    print(f'detection {j} encountered problem')
                    import code
                    code.interact(local=dict(globals(), **locals()))
            
            new_tree.write(self.output_file, encoding='utf-8', xml_declaration=True) #save as progress incase of crash
            print(len(sliced_detections),'masks converted in image',image_name)


        new_tree.write(self.output_file, encoding='utf-8', xml_declaration=True)
        zip_filename = self.output_file.split('.')[0] + '.zip'
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.output_file, arcname='output_xml_file.xml')
        print('XML file zipped')


# print("Detecting corals and saving to annotation format")
# Det = Predict2Cvat(base_img_location, output_filename, weight_file, base_file, save_img=False, max_img=max_img)
# Det.run()
# print("Done detecting corals")

# import code
# code.interact(local=dict(globals(), **locals()))

import cv2 as cv
def save_image_predictions_mask(results, image, imgname, save_path, conf, class_list, classes, class_colours):
    """save_image_predictions_mask
    saves the predicted masks results onto an image, recoring confidence and class as well. 
    Can also show ground truth anotiations from the associated textfile (assumed annotiations are normalised xy corifinate points)
    """
    # ## to see image as 640 resolution
    # image = cv.imread(imgname)
    # image = cv.resize(image, (640, 488))
    height, width, _ = image.shape
    masked = image.copy()
    line_tickness = int(round(width)/600)
    font_size = 2#int(round(line_tickness/2))
    font_thickness = 5#3*(abs(line_tickness-font_size))+font_size
    if results:
        for j, m in enumerate(results):
            contours, _ = cv2.findContours(m[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = np.squeeze(contour)
            cls = classes[int(class_list[j])]
            desired_color = class_colours[cls]
            if points is None or not points.any() or len(points) == 0:
                print(f'mask {j} encountered problem with points {points}, class is {cls}')
            else: 
                cv.fillPoly(masked, [points], desired_color) #here the polygons are wrong
                xmin = min(points[:, 1])
                ymin = min(points[:, 0])
                #cv.putText(image, f"{conf[j]:.2f}: {cls}", (int(xmin-20), int(ymin - 5)), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
        for t, b in enumerate(results.xyxy):
            cls = classes[int(class_list[t])]
            desired_color = class_colours[cls]
            cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), tuple(class_colours[cls]), line_tickness)
            cv.putText(image, f"{conf[t]:.2f}: {cls}", (int(b[0]-20), int(b[1] - 5)), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    else:
        print(f'No masks found in {imgname}')

    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det_mask.jpg')
    cv.imwrite(imgsave_path, semi_transparent_mask)
    import code
    code.interact(local=dict(globals(), **locals()))
    

## Visualise Detections on images
if visualise:
    print("visulising detections")
    os.makedirs(save_dir, exist_ok=True)
    imglist = sorted(glob.glob(os.path.join(base_img_location, '*.jpg')))
    for i, image_file in enumerate(imglist):
        print(f"processing image: {i+1} of {len(imglist)}")
        if i>max_img:
            print("Hit max img limit")
            break
        image = cv2.imread(image_file)
        image_height, image_width = image.shape[:2]
        data_dict = {'class_name': []}
        box_list, conf_list, cls_id_list, mask_list = [], [], [], []
        whole_image_mask = np.zeros((image_height, image_width), dtype=bool)
        print("Batching image")
        for y in range(0, image_height, batch_height):
            for x in range(0, image_width, batch_width):
                y_end = min(y + batch_height, image_height)
                x_end = min(x + batch_width, image_width)
                img= image[y:y_end, x:x_end]
                sliced_detections = slicer(image=img)
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
                    mask_resized = cv2.resize(mask.astype(np.uint8), (batch_width, batch_height))
                    mask_list.append(mask_resized)
                whole_image_mask[y:y_end, x:x_end] = 0
        print("stiching batch back together")
        whole_image_detection = sv.Detections(xyxy=np.array(box_list), confidence=np.array(conf_list), class_id=np.array(cls_id_list), 
                                              mask=np.array(mask_list), data=data_dict)
        save_image_predictions_mask(whole_image_detection, image, image_file, save_dir, conf_list, cls_id_list, classes, class_colours)
        import code
        code.interact(local=dict(globals(), **locals()))
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=whole_image_detection) #this takes long time (>1h)
        #sv.plot_image(annotated_image) #visualise all the detections    
        cv2.imwrite(f"{os.path.join(save_dir,os.path.basename(image_file)[:-4])}_det.jpg", annotated_image)


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

