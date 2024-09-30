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
import supervision as sv

ultralitics_version = False #set to true, if want example of ultralitics prediction
SAHI = True #set to true if using SAHI to look at 640p images
batch = True #image too big / too many predictions to do in one go
batch_height, batch_width = 3000, 3000
#weights_file_path = '/home/java/Java/ultralytics/runs/segment/train6/weights/best.pt' #trained on tilled images
#weights_file_path = '/home/java/Java/ultralytics/runs/segment/train4/weights/best.pt' #trained on 640 imgsz
# weights_file_path = '/home/java/Java/ultralytics/runs/segment/train5/weights/best.pt' #trained on 1280 imgsz
#weight_file = "/home/java/Java/ultralytics/runs/segment/train9/weights/cgras_yolov8n-seg_640p_20231209.pt" #dorian used
weights_file_path = "/home/java/Java/ultralytics/runs/segment/train21/weights/best.pt" #trained on 640 imgsz dataset combined 22 and 23

save_dir = '/media/java/CGRAS-SSD/cgras_23_n_24_combined/split_24_09_19/visualise'
img_folder = os.path.join('/media/java/CGRAS-SSD/cgras_23_n_24_combined/split_24_09_19/test/images')
#txt_folder = os.path.join(save_dir, 'train', 'labels')
txt_folder = os.path.join('/media/java/CGRAS-SSD/cgras_23_n_24_combined/split_24_09_19/test/labels')

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

def plot_ground_truth(image, txt, classes, class_colours, line_tickness):
    """plot_ground_truth
    plots the ground truth annotiations onto an image
    """
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
    return image

def save_image_predictions_mask(results, image, imgname, save_path, conf, class_list, classes, class_colours, ground_truth=False, txt=None):
    """save_image_predictions_mask
    saves the predicted masks results onto an image, recoring confidence and class as well. 
    Can also show ground truth anotiations from the associated textfile (assumed annotiations are normalised xy corifinate points)
    """
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
            cls_name = classes[int(class_list[j])]
            desired_color = class_colours[cls_name]
            if points is None or not points.any() or len(points) == 0:
                print(f'mask {j} encountered problem with points {points}, class is {cls_name}')
            else: 
                cv.fillPoly(masked, [points], desired_color)
                xmin = min(xyn[0, :, 0])
                ymin = min(xyn[0, :, 1])
                cv.putText(image, f"{conf[j]:.2f}: {cls_name}", (int(xmin-20), int(ymin - 5)), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    else:
        print(f'No masks found in {imgname}')
    
    if ground_truth & (txt is not None):
        image = plot_ground_truth(image, txt, classes, class_colours, line_tickness)

    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det_mask.jpg')
    cv.imwrite(imgsave_path, semi_transparent_mask)

    # import code
    # code.interact(local=dict(globals(), **locals()))

def is_overlap_box(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    if x1 > x4 or x3 > x2:
        return False
    if y1 > y4 or y3 > y2:
        return False
    return True

def create_overlapped(box1, box2, i, j, mask1, mask2, conf_list, cls_id_list):
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
    return new_box, new_conf, new_class, new_mask, new_tl_x, new_tl_y, new_w, new_h

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
            if is_overlap_box(box1, box2): #assume only pairs overlap
                overlap = True
                overlap_count += 1
                new_box, new_conf, new_class, new_mask, new_tl_x, new_tl_y, new_w, new_h = create_overlapped(
                    box1, box2, i, j, mask1, mask2, conf_list, cls_id_list)
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

def callback(image_slice: np.ndarray) -> sv.Detections:
    results = model(image_slice)
    try:
        detections = sv.Detections.from_ultralytics(results[0])
    except:
        print("Error in callback")
        import code
        code.interact(local=dict(globals(), **locals()))
    return detections

def save_img_batch(image_cv, box_array, conf_list, cls_id_list, mask_list, image_name, save_dir):
    height, width, _ = image_cv.shape
    masked = image_cv.copy()
    line_tickness = int(round(width)/600)
    font_size = 2#int(round(line_tickness/2))
    font_thickness = 5#3*(abs(line_tickness-font_size))+font_size
    if conf_list is not None:
        for j, m in enumerate(mask_list):
            contours, _ = cv.findContours(m[0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
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
                    cv.fillPoly(masked, [points], desired_color) 
        for t, b in enumerate(box_array):
            cls = classes[int(cls_id_list[t])]
            desired_color = class_colours[cls]
            cv.putText(image_cv, f"{conf_list[t]:.2f}: {cls}", (int(b[0]-20), int(b[1] - 5)), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    else:
        print(f'No masks found in {image_name}')
    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image_cv, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.basename(image_name)
    imgsave_path = os.path.join(save_dir, imgsavename[:-4] + '_det_mask.jpg')
    cv.imwrite(imgsave_path, semi_transparent_mask)

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_file_path).to(device)
slicer = sv.InferenceSlicer(callback=callback, slice_wh=(640, 640), overlap_ratio_wh=(0.1, 0.1))

print('Model Inference:')

imglist = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
txtlist = sorted(glob.glob(os.path.join(txt_folder, '*.txt')))
imgsave_dir = save_dir
os.makedirs(imgsave_dir, exist_ok=True)

# if not ultralitics_version:
#     for i, imgname in enumerate(imglist):
#         print(f'predictions on {i+1}/{len(imglist)}')
#         if i >= 10: # for debugging purposes
#             break 
#             import code
#             code.interact(local=dict(globals(), **locals()))
#         image = cv.imread(imgname)
#         results = model.predict(source=imgname, iou=0.5, agnostic_nms=True, imgsz=640)
#         conf, class_list = [], [] 
#         for j, b in enumerate(results[0].boxes):
#             conf.append(b.conf.item())
#             class_list.append(b.cls_name.item())
        
#         txt = txtlist[i]
#         ground_truth = True
#         save_image_predictions_mask(results, image, imgname, imgsave_dir, conf, class_list, classes, class_colours, ground_truth, txt)

if SAHI:
    for i, imgname in enumerate(imglist):
        print(f'SHAI predictions on {i+1}/{len(imglist)}')
        if i >= 3: # for debugging purposes
            print("Hit Max img")
            import code
            code.interact(local=dict(globals(), **locals()))
            break
        image = cv.imread(imgname)
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
                        mask_resized = cv.resize(mask.astype(np.uint8), (x_end - x, y_end - y))
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
            save_img_batch(image, box_array, conf_list, cls_id_list, mask_list, os.path.basename(imgname), save_dir)
            save_img_batch(image, updated_box_array, updated_conf_list, updated_class_id, updated_mask_list, os.path.basename(imgname), save_dir)
        else:
            sliced_detections = slicer(image=image)
            masked = image.copy()
            for detection in sliced_detections:
                xyxy = detection[0].tolist()
                mask_array = detection[1].astype(np.uint8) 
                confidence = detection[2]
                class_id = detection[3]
                contours, _ = cv.findContours(mask_array, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    points = np.squeeze(contour)
                    if len(points.shape) == 1:
                        points = points.reshape(-1, 1, 2)
                    elif len(points.shape) == 2 and points.shape[1] != 2:
                        points = points.reshape(-1, 1, 2)
                    cls_name = classes[class_id]
                    desired_color = class_colours[cls_name]
                    if points is None or not points.any() or len(points) == 0:
                        print(f'mask encountered problem with points {points}, class is {cls_name}')
                    else: 
                        cv.fillPoly(masked, [points], desired_color) 
                cv.putText(image, f"{confidence:.2f}: {cls_name}", (int(xyxy[0]-20), int(xyxy[1] - 5)), cv.FONT_HERSHEY_SIMPLEX, 2, desired_color, 5)
            alpha = 0.30
            semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
            imgsave_path = os.path.join(save_dir, os.path.basename(image_file)[:-4] + '_t1.jpg')
            cv.imwrite(imgsave_path, semi_transparent_mask)


import code
code.interact(local=dict(globals(), **locals()))



if ultralitics_version: #ultralytics code
    for i, imgname in enumerate(imglist):
        if i >= 10: # for debugging purposes
            break 
            import code
            code.interact(local=dict(globals(), **locals()))
        image = cv.imread(imgname)
        results = model.predict(source=image, iou=0.5, agnostic_nms=True)
        conf = []
        class_list = []
        #masks = results[0].masks
        for i, b in enumerate(results[0].boxes):
            conf.append(b.conf.item())
            class_list.append(b.cls_name.item())
            #seg = masks[i].xyn[0]
        for r in results:
            im_array = r.plot(conf=True, line_width=4, font_size=4, boxes=True)
            im = Image.fromarray(im_array[..., ::-1])
            im.show()
        r.save_txt(image_file[:-4]+'_'+ '_detmask.txt', save_conf=True ) 
            ## Saves masks like: (class being first number and confidence being the last, middle numbers the mask.xyn values ) masks[j].xyn[0].copy().reshape(-1)
            ## 9 0.207813 0.368893 0.20625 0.371234 0.20625 0.392305 0.207813 0.394646 0.215625 0.394646 0.217187 0.392305 0.217187 0.389963 0.220313 0.385281 0.221875 0.385281 0.223438 0.38294 0.223438 0.371234 0.221875 0.368893 0.971601
        

# for interactive debugger in terminal:
import code
code.interact(local=dict(globals(), **locals()))
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
            cls_name = classes[int(b.cls_name.item())]
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