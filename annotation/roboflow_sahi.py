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

weight_file = '/home/java/Java/ultralytics/runs/segment/train9/weights/cgras_yolov8n-seg_640p_20231209.pt'
base_img_location = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/ultralytics_data'
save_dir = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/ultralytics_data_detections'
base_file = "/home/java/Downloads/emptyCGRAS2023/annotations.xml"
output_filename = "/home/java/Downloads/cgras_2024_complete.xml"
max_img = 2
single_image = True #run roboflow sahi on one image and get detected segmentation results
visualise = False #visualise the detections on the images

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
    flat_mask = np.concatenate([flat_mask, [0]])
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
slicer = sv.InferenceSlicer(callback=callback)

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
        if save_img:
            self.save_dir = save_dir


    def run(self):
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


        for i, image_element in enumerate(root.findall('.//image')):
            print(i+1,'images being processed')
            if i>self.max_img:
                print("Hit max img limit")
                break
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
            
            if i>30:
                continue
            image_file = os.path.join(self.img_location, image_name)
            image_cv = cv2.imread(image_file)
            sliced_detections = slicer(image=image_cv)

            if self.save_img:
                annotated_image = mask_annotator.annotate(scene=image_cv.copy(), detections=sliced_detections)
                cv2.imwrite(f"{os.path.join(self.save_dir,os.path.basename(image_file)[:-4])}_det.jpg", annotated_image)

            if sliced_detections is None:
                print('No masks found in image',image_name)
                continue

            for j, detection in enumerate(sliced_detections):
                try:
                    xyxy = detection[0].tolist()
                    mask_array = detection[1]
                    rle = binary_mask_to_rle(mask_array) 
                    rle_string = ', '.join(map(str, rle))
                    left, top, width, height = min(xyxy[0], xyxy[2]), min(xyxy[1], xyxy[3]), abs(xyxy[0] - xyxy[2]), abs(xyxy[1] - xyxy[3])
                    label = detection[5]['class_name']
                    mask_elem = SubElement(new_elem, 'mask')
                    mask_elem.set('label', label)
                    mask_elem.set('source', 'semi-auto')
                    mask_elem.set('occluded', '0')
                    mask_elem.set('rle', rle_string)
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
# Det = Predict2Cvat(base_img_location, output_filename, weight_file, base_file, save_img=False)
# Det.run()
# print("Done detecting corals")

# import code
# code.interact(local=dict(globals(), **locals()))

## Visualise Detections on images
if visualise:
    os.makedirs(save_dir, exist_ok=True)
    imglist = sorted(glob.glob(os.path.join(base_img_location, '*.jpg')))
    for i, image_file in enumerate(imglist):
        print(f"processing image: {i+1} of {len(imglist)}")
        if i>max_img:
            print("Hit max img limit")
            break
        image = cv2.imread(image_file)
        sliced_detections = slicer(image=image)
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=sliced_detections)

        #sv.plot_image(annotated_image) #visualise all the detections    
        cv2.imwrite(f"{os.path.join(save_dir,os.path.basename(image_file)[:-4])}_det.jpg", annotated_image)


### Single image check and test
if single_image:
    image_file = "/home/java/Java/data/cgras_20231028/images/756_20211213_427.jpg"
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

