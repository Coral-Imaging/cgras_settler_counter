#! /usr/bin/env python3

"""test_sahi.py
seeing if the sahi works fro yolo8 https://docs.ultralytics.com/guides/sahi-tiled-inference/
possible segmetation: https://github.com/obss/sahi/pull/918/files
"""

# pip install -U ultralytics sahi ##TODO add to yaml file for making enviroment

from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import visualize_object_predictions
from PIL import Image
import cv2 as cv
from numpy import asarray
import os
from Utils import classes, class_colours
import numpy as np
from sahi.slicing import slice_coco
from ultralytics.data.converter import convert_coco
import shutil
import glob
from Utils import classes, poly_2_rle
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
import zipfile

######################## spliting images ########################################
# Download annotations from cvat using coco fromat (cvat should have images in polygons)
# then use slice coco to cut the images up and the annotations accordingly
# then convert to yolo version for future use
#################################################################################

# output_path = '/home/java/Java/data/cgras_20230421/sahi'
# coco_ann_file = '/home/java/Java/data/cgras_20230421/coco/annotations/instances_default.json'
# coco_ann_save_name = 'test'

# yolo_save = os.path.join(output_path, 'train')

# coco_dict, coco_path = slice_coco(
#     coco_annotation_file_path=coco_ann_file,
#     image_dir='/home/java/Java/data/cgras_20230421/train/images',
#     output_coco_annotation_file_name=coco_ann_save_name,
#     output_dir=output_path,
#     slice_height=640,
#     slice_width=640,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2,
#     min_area_ratio=0.1,
#     ignore_negative_samples=True
# )
# print("done cutting")

# import code
# code.interact(local=dict(globals(), **locals()))

# convert_coco(labels_dir=output_path, save_dir=yolo_save,
#                  use_segments=True, use_keypoints=False, cls91to80=False)

# lable_dir = os.path.join(yolo_save, 'labels')
# coco_labels = os.path.join(lable_dir, coco_ann_save_name+'_coco')
# for filename in os.listdir(coco_labels):
#     source = os.path.join(coco_labels, filename)
#     destination = os.path.join(lable_dir, filename)
#     if os.path.isfile(source):
#         shutil.move(source, destination)

# for filename in os.listdir(output_path):
#     source = os.path.join(output_path, filename)
#     destination = os.path.join(yolo_save, 'images', filename)
#     if os.path.isfile(source):
#         shutil.move(source, destination)

# print("files in yolo format and directory")

# import code
# code.interact(local=dict(globals(), **locals()))

################## Detection code ##############################
print("running detection code")

def save_image_predictions_bb(predictions, imgname, imgsavedir, class_colours, classes):
    """
    save predictions/detections (assuming predictions in yolo format) on image as bounding box
    """
    img = cv.imread(imgname)
    imgw, imgh = img.shape[1], img.shape[0]
    for p in predictions:
        if isinstance(p[0:4], list):
            x1, y1, x2, y2 = p[0:4]
        else:
            x1, y1, x2, y2 = p[0:4].tolist()
        conf = p[4]
        cls = int(p[5])
        #extract back into cv lengths
        x1 = x1*imgw
        x2 = x2*imgw
        y1 = y1*imgh
        y2 = y2*imgh        
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), class_colours[classes[cls]], 4)
        cv.putText(img, f"{classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 1.5, class_colours[classes[cls]], 4)
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '_det.jpg')
    cv.imwrite(imgsave_path, img)
    return True

# Download YOLOv8 model
yolov8_model_path = '/home/java/Java/ultralytics/runs/segment/train6/weights/best.pt' #trained on tilled images
#yolov8_model_path = '/home/java/Java/ultralytics/runs/segment/train5/weights/best.pt' #trained on 1280 imgsz
export_dir="/home/java/Java/data/cgras_20230421/sahi"
download_yolov8s_model(yolov8_model_path)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    mask_threshold=0.3,
    confidence_threshold=0.3,
    device="cuda:0" # or "cpu" 
)

image_file_list = sorted(glob.glob(os.path.join('/home/java/Java/data/cgras_20230421/train/images', '*.jpg')))
## testing and playing with functions
def detection_test(image_file_list, export_dir, detection_model):
    for i, image_file in enumerate(image_file_list):
        if i>10:
            break
        img = cv.imread(image_file)
        imgw, imgh = img.shape[1], img.shape[0] 

        ##Prediction - working
        result = get_prediction(image_file, detection_model)
        result.export_visuals(export_dir=export_dir)
        object_prediction_list = result.object_prediction_list
        predictions = []
        for obj in object_prediction_list:
            cls = obj.category.id
            conf = obj.score.value
            bb = obj.bbox
            x1, y1, x2, y2 = bb.minx, bb.miny, bb.maxx, bb.maxy
            x1n, y1n, x2n, y2n = x1/imgw, y1/imgh, x2/imgw, y2/imgh
            predictions.append([x1n, y1n, x2n, y2n, conf, cls])
        save_image_predictions_bb(predictions, image_file, export_dir, class_colours, classes)

        # import code
        # code.interact(local=dict(globals(), **locals()))

        ##Sliced inference - 
        # working but has lots of False positives that are small boxes around random stuff
        result = get_sliced_prediction(
            image_file,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1
        )

        object_prediction_list = result.object_prediction_list
        predictions = []
        for obj in object_prediction_list:
            cls = obj.category.id
            conf = obj.score.value
            bb = obj.bbox
            x1, y1, x2, y2 = bb.minx, bb.miny, bb.maxx, bb.maxy
            x1n, y1n, x2n, y2n = x1/imgw, y1/imgh, x2/imgw, y2/imgh
            predictions.append([x1n, y1n, x2n, y2n, conf, cls])

        # import code
        # code.interact(local=dict(globals(), **locals()))
        save_image_predictions_bb(predictions, image_file, export_dir, class_colours, classes)

        ##saves the image using sahi / yolo visualisation results, will have the same detections as above
        img_converted = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        numpydata = asarray(img_converted)
        visualize_object_predictions(
            numpydata, 
            object_prediction_list = result.object_prediction_list,
            hide_labels = 1, 
            output_dir=export_dir,
            file_name = 'vis_result',
            export_format = 'png'
        )
        #does the same as visualise_object_predictions
        result.export_visuals(export_dir=export_dir) #

# detection_test(image_file_list, export_dir, detection_model)
# import code
# code.interact(local=dict(globals(), **locals()))

############## Pretend human in the loop ##############
print("pretend human in the loop")
#folder of new images
img_location = os.path.join('/home/java/Java/data/cgras_20230421/train','images')
#images were in cvat and then downloaded in CVAT format
base_ann_file = "/home/java/Downloads/dec14/annotations.xml"
output_file = "/home/java/Downloads/dec14_box.xml"

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
        result = get_sliced_prediction(
            image_file,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1
        )

        object_prediction_list = result.object_prediction_list
        predictions = []
        for j, obj in enumerate(object_prediction_list):
            cls = obj.category.id
            bb = obj.bbox
            x1, y1, x2, y2 = bb.minx, bb.miny, bb.maxx, bb.maxy
            label = classes[int(cls)]
            if x1 is None:
                print(f'mask {j} encountered problem box = {bb}')
                import code
                code.interact(local=dict(globals(), **locals()))
            else:
                mask_elem = SubElement(new_elem, 'box')
                mask_elem.set('label', label)
                mask_elem.set('source', 'manual')
                mask_elem.set('occluded', '0')
                mask_elem.set('xtl', str(x1))
                mask_elem.set('ytl', str(y1))
                mask_elem.set('xbr', str(x2))
                mask_elem.set('ybr', str(y2))
                mask_elem.set('z_order', '0')
        print(len(object_prediction_list),'masks converted in image',image_name)

    new_tree.write(output_file, encoding='utf-8', xml_declaration=True)

    zip_filename = output_file.split('.')[0] + '.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname='output_xml_file.xml')
    print('XML file zipped')

    import code
    code.interact(local=dict(globals(), **locals()))