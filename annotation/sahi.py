#! /usr/bin/env python3

"""test_sahi/py
seeing if the sahi works fro yolo8 https://docs.ultralytics.com/guides/sahi-tiled-inference/
pssible segmetation: https://github.com/obss/sahi/pull/918/files
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
yolov8_model_path = "/home/java/Java/ultralytics/runs/segment/train4/weights/best.pt"
export_dir="/home/java/Java/data/cgras_20230421/detections/sahi"
download_yolov8s_model(yolov8_model_path)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    mask_threshold=0.3,
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0'
)

image_file = '/home/java/Java/data/cgras_20230421/train/images/00_20230116_MIS1_RC_Aspat_T04_08.jpg'
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


import code
code.interact(local=dict(globals(), **locals()))

##Sliced inference - working but has lots of FP
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

result.export_visuals(export_dir=export_dir)

save_image_predictions_bb(predictions, image_file, export_dir, class_colours, classes)

import code
code.interact(local=dict(globals(), **locals()))


