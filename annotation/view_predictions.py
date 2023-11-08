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

ultralitics_version = False #set to true, if want example of ultralitics prediction
weights_file_path = '/home/java/Java/ultralytics/runs/segment/train4/weights/best.pt'
save_dir = '/home/java/Java/data/cgras_20230421'
img_folder = '/home/java/Java/data/cgras_20230421/train/images'
txt_folder = '/home/java/Java/data/cgras_20230421/train/labels'


#save txt results like they would be saved by ultralytics
def save_txt_predictions_masks(results, conf, class_list, save_path):
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

def save_image_predictions_mask(results, image, imgname, save_path, class_list, classes, class_colours, ground_truth, txt):
    height, width, _ = image.shape
    masked = image.copy()
    line_tickness = int(round((height/width)*4))
    font_size = int(round(line_tickness/2))
    font_thickness = 3*(abs(line_tickness-font_size))+font_size
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
            cv.putText(image, f"{cls}: {conf[j]:.2f}", (int(xmin-20), int(ymin - 5)), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    
    if ground_truth:
        points_normalised, points, class_idx = [], [], []
        with open(txt, "r") as file:
            lines = file.readlines()
        for line in lines:
                data = line.strip().split()
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
            cv.polylines(image, [pointers], True, class_colours[classes[class_idx[idx]]], line_tickness)

    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det.jpg')
    cv.imwrite(imgsave_path, semi_transparent_mask)

    import code
    code.interact(local=dict(globals(), **locals()))

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_file_path).to(device)


# get predictions
print('Model Inference:')


imglist = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
txtlist = sorted(glob.glob(os.path.join(txt_folder, '*.txt')))
imgsave_dir = os.path.join(save_dir, 'detections', 'detections_images')
os.makedirs(imgsave_dir, exist_ok=True)
for i, imgname in enumerate(imglist):
    print(f'predictions on {i+1}/{len(imglist)}')
    if i >= 5: # for debugging purposes
        break
    image = cv.imread(imgname)
    results = model.predict(source=imgname, iou=0.5, agnostic_nms=True)
    conf, class_list = [], [], 
    for j, b in enumerate(results[0].boxes):
        conf.append(b.conf.item())
        class_list.append(b.cls.item())
    imgsavename = os.path.basename(imgname)
    
    txt = txtlist[i]
    ground_truth = True
    save_image_predictions_mask(results, image, imgname, imgsave_dir, class_list, classes, class_colours, ground_truth, txt)


# for interactive debugger in terminal:
import code
code.interact(local=dict(globals(), **locals()))


image_file = '/home/java/Java/data/cgras_20230421/train/images/00_20230116_MIS1_RC_Aspat_T04_08.jpg'
if ultralitics_version: #ultralytics code
    image = cv.imread(image_file)
    results = model.predict(source=image_file, iou=0.5, agnostic_nms=True)
    conf = []
    class_list = []
    masks = results[0].masks
    for i, b in enumerate(results[0].boxes):
        conf.append(b.conf.item())
        class_list.append(b.cls.item())
        seg = masks[i].xyn[0]
    for r in results:
        im_array = r.plot(conf=True, line_width=4, font_size=4, boxes=True)
        im = Image.fromarray(im_array[..., ::-1])
        im.show()
    #r.save_txt(image_file[:-4]+'_'+ '_detmask.txt', save_conf=True ) 
        ### Saves masks like: (class being first number and confidence being the last, middle numbers the mask.xyn values ) masks[j].xyn[0].copy().reshape(-1)
        ### 9 0.207813 0.368893 0.20625 0.371234 0.20625 0.392305 0.207813 0.394646 0.215625 0.394646 0.217187 0.392305 0.217187 0.389963 0.220313 0.385281 0.221875 0.385281 0.223438 0.38294 0.223438 0.371234 0.221875 0.368893 0.971601
    import code
    code.interact(local=dict(globals(), **locals()))

#not working old code (08/11/2023) that trys to use the mask aspect of the mask. Mask are vertically shifted 
def save_image_predictions_mask_old(results, image, imgname, save_path, class_list, classes, class_colours, ground_truth, txt):
    m_idx_list = []
    height, width, _ = image.shape
    masked = image.copy()
    for m in results[0].masks:
        mask_raw = m.cpu().data.numpy().transpose(1, 2, 0)
        mask_3channel = cv.merge((mask_raw,mask_raw,mask_raw)) 
        mask = cv.resize(mask_3channel, (width, height), interpolation=cv.INTER_LINEAR) 
        mask_indices = np.where(mask != 0)
        m_idx_list.append(mask_indices)

    for i, j in enumerate(m_idx_list):
        desired_color = class_colours[classes[int(class_list[i])]]
        masked[j[0], j[1]] = desired_color

    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det.jpg')
    cv.imwrite(imgsave_path, semi_transparent_mask)

    import code
    code.interact(local=dict(globals(), **locals()))