#! /usr/bin/env/python3

"""
quick bit of code to visualise the predition results using a trained yolov8 weights file (ideally from a trained run)
"""

from ultralytics import YOLO
import os
import glob
import torch
import cv2 as cv
import numpy as np
from PIL import Image
from Utils import classes, class_colours

ultralitics_version = True #set to true, if want example of ultralitics prediction
#weights_file_path = '/home/java/Java/ultralytics/runs/segment/train6/weights/best.pt' #trained on tilled images
#weights_file_path = '/home/java/Java/ultralytics/runs/segment/train4/weights/best.pt' #trained on 640 imgsz
# weights_file_path = '/home/java/Java/ultralytics/runs/segment/train5/weights/best.pt' #trained on 1280 imgsz
# weight_file = "/home/java/Java/ultralytics/runs/segment/train9/weights/cgras_yolov8n-seg_640p_20231209.pt" #dorian used?
weights_file_path = '/home/java/Java/ultralytics/runs/detect/EGH455-TID---Gauge-Fix-1 train12/weights/best.pt' 
save_dir = '/home/java/Java/datasets/EGH455-TID---Gauge-Fix-1/'
img_folder = os.path.join('/home/java/Java/datasets/EGH455-TID---Gauge-Fix-1/test/images')
#txt_folder = os.path.join(save_dir, 'train', 'labels')
txt_folder = os.path.join('/home/java/Java/datasets/EGH455-TID---Gauge-Fix-1/test/labels')

#save txt results like they would be saved by ultralytics
def save_txt_predictions_masks(results, conf, class_list, save_path):
    """save_txt_predictions_masks
    writes a textfile for the segmentation results, saving the data as ultralytics does
    """
    masks = results[0].masks
    txt_results = []
    for i, r in enumerate(masks):
        txt_result1 = [int(class_list[i])]
        seg = masks[i].xyn[0].copy().reshape(-1)
        for j, s in enumerate(seg):
            txt_result1.append(seg[j])
        txt_result1.append(conf[i])
        txt_results.append(txt_result1)
    with open(save_path, 'w') as file:
        for txt_result in txt_results:
            for item in txt_result:
                file.write(str(item) + ' ')
            file.write('\n')

def save_image_predictions_mask(results, image, imgname, save_path, conf, class_list, classes, class_colours, ground_truth=False, txt=None):
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
    font_size = 1#int(round(line_tickness/2))
    font_thickness = 1#3*(abs(line_tickness-font_size))+font_size
    if results and results[0].masks:
        for j, m in enumerate(results[0].masks):
            xyn = np.array(m.xyn)
            xyn[0, :, 0] = (xyn[0, :, 0] * width)
            xyn[0, :, 1] = (xyn[0, :, 1] * height)
            points = xyn.reshape((-1, 1, 2)).astype(np.int32)
            cls = classes[int(class_list[j])]
            desired_color = class_colours[cls]
            if points is None or not points.any() or len(points) == 0:
                print(f'mask {j} encountered problem with points {points}, class is {cls}')
            else: 
                cv.fillPoly(masked, [points], desired_color)
                xmin = min(xyn[0, :, 0])
                ymin = min(xyn[0, :, 1])
                cv.putText(image, f"{conf[j]:.2f}: {cls}", (int(xmin-20), int(ymin - 5)), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    else:
        print(f'No masks found in {imgname}')
    
    if ground_truth & (txt is not None):
        points_normalised, points, class_idx = [], [], []
        with open(txt, "r") as file:
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
                print(f'invalid line there is {len(data)} data, related to img {imgname}')
        for idx in range(len(class_idx)):
            pointers = np.array(points[idx], np.int32).reshape(-1,2)
            cv.polylines(image, [pointers], True, class_colours[classes[class_idx[idx]]], line_tickness)

    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det_mask.jpg')
    cv.imwrite(imgsave_path, semi_transparent_mask)

    # import code
    # code.interact(local=dict(globals(), **locals()))

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_file_path).to(device)

print('Model Inference:')

imglist = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
txtlist = sorted(glob.glob(os.path.join(txt_folder, '*.txt')))
imgsave_dir = save_dir
os.makedirs(imgsave_dir, exist_ok=True)

if not ultralitics_version:
    for i, imgname in enumerate(imglist):
        print(f'predictions on {i+1}/{len(imglist)}')
        if i >= 10: # for debugging purposes
            break 
            import code
            code.interact(local=dict(globals(), **locals()))
        image = cv.imread(imgname)
        results = model.predict(source=imgname, iou=0.5, agnostic_nms=True, imgsz=640)
        conf, class_list = [], [] 
        for j, b in enumerate(results[0].boxes):
            conf.append(b.conf.item())
            class_list.append(b.cls.item())
        
        #txt = txtlist[i]
        ground_truth = False
        save_image_predictions_mask(results, image, imgname, imgsave_dir, conf, class_list, classes, class_colours)


# if ultralitics_version: #ultralytics code
#     for i, imgname in enumerate(imglist):
#         if i >= 10: # for debugging purposes
#             break 
#             import code
#             code.interact(local=dict(globals(), **locals()))
#         image = cv.imread(imgname)
#         results = model.predict(source=image, iou=0.5, agnostic_nms=True)
#         conf = []
#         class_list = []
#         #masks = results[0].masks
#         for i, b in enumerate(results[0].boxes):
#             conf.append(b.conf.item())
#             class_list.append(b.cls.item())
#             #seg = masks[i].xyn[0]
#         for r in results:
#             im_array = r.plot(conf=True, line_width=4, font_size=4, boxes=True)
#             im = Image.fromarray(im_array[..., ::-1])
#             im.show()
        #r.save_txt(image_file[:-4]+'_'+ '_detmask.txt', save_conf=True ) 
            ### Saves masks like: (class being first number and confidence being the last, middle numbers the mask.xyn values ) masks[j].xyn[0].copy().reshape(-1)
            ### 9 0.207813 0.368893 0.20625 0.371234 0.20625 0.392305 0.207813 0.394646 0.215625 0.394646 0.217187 0.392305 0.217187 0.389963 0.220313 0.385281 0.221875 0.385281 0.223438 0.38294 0.223438 0.371234 0.221875 0.368893 0.971601
        

# for interactive debugger in terminal:
print("for bounding boxes")


def save_image_predictions(results, image, imgname, save_path, classes, class_colours, ground_truth=False, txt=None):
    """save_image_predictions
    saves the predicted bb results onto an image, recoring confidence and class as well. 
    Can also show ground truth anotiations from the associated textfile (assumed annotiations are normalised xy corifinate points)
    """
    height, width, _ = image.shape
    line_tickness = int(round(width)/600)
    font_size = 1#int(round(line_tickness/2))
    font_thickness = 1#3*(abs(line_tickness-font_size))+font_size
    if results and results[0].boxes:
        for j, b in enumerate(results[0].boxes):
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_name = classes[int(b.cls.item())]
            desired_color = class_colours[cls_name]
            conf = b.conf.item()
            cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2),), desired_color, line_tickness*2)
            cv.putText(image, f"{conf:.2f}: {cls_name}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    else:
        print(f'No boxes found in {imgname}')
    if ground_truth & (txt is not None):
        with open(txt, "r") as file:
            lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            cls_name = classes[int(data[0])]
            desired_color = class_colours[cls_name]
            x1 = int(float(data[1])*width)
            y1 = int(float(data[2])*height)
            w = int(float(data[3])*width)
            h = int(float(data[4])*height)
            cv.rectangle(image, (x1-w, y1-h), (x1+h, y1+w), desired_color, line_tickness)
            cv.putText(image, f"{cls_name}", (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    #VID_20230814_151633_mp4-100_jpg.rf.26716244af5572980b7b449a0bf170a6_det
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det.jpg')
    cv.imwrite(imgsave_path, image)


for i, imgname in enumerate(imglist):
        print(f'predictions on {i+1}/{len(imglist)}')
        if i >= 10: # for debugging purposes
            break 
            import code
            code.interact(local=dict(globals(), **locals()))
        image = cv.imread(imgname)
        results = model.predict(source=imgname, iou=0.5, agnostic_nms=True, imgsz=640)
        
        
        txt = txtlist[i]
        ground_truth = True
        save_image_predictions(results, image, imgname, imgsave_dir, classes, class_colours,  ground_truth, txt)