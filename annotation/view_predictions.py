#! /usr/bin/env/python3

"""
quick bit of code to visualise the predition results using a trained yolov8 weights file (ideally from a trained run)
"""

from ultralytics import YOLO
import os
import glob
import torch
import cv2 as cv

# save_dir = '/home/java/Java/ultralytics/runs/segment/train3'
weights_file_path = '/home/java/Java/ultralytics/runs/segment/train3/weights/best.pt'
# image_file = '/home/java/Java/data/cgras_20230421/train/images/00_20230116_MIS1_RC_Aspat_T04_08.jpg'
# img_folder ='some/path/to/a/bunch/of/images'
# multiple_img = False
save_dir = '/home/java/Java/data/cgras_20231028'
img_folder = '/home/java/Java/data/cgras_20231028/images'
multiple_img = True


classes = ['recruit_live_white', 'recruit_cluster_live_white', 'recruit_symbiotic', 'recruit_symbiotic_cluster', 'recruit_partial',
           'recruit_partial_dead', 'recruit_dead', 'recruit_cluster_dead', 'unknown', 'pest_tubeworm']

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
class_colours = {classes[0]: blue,
                classes[1]: green,
                classes[2]: purple,
                classes[3]: yellow,
                classes[4]: brown,
                classes[5]: cyan,
                classes[6]: orange,
                classes[7]: red,
                classes[9]: light_grey,
                classes[8]: dark_purple}


def save_image_predictions(predictions, imgname, imgsavedir, class_colours, classes):
    """
    save predictions/detections (assuming predictions in yolo format) on image
    """
    img = cv.imread(imgname)
    imgw, imgh = img.shape[1], img.shape[0]
    for p in predictions:
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

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_file_path).to(device)

# get predictions
print('Model Inference:')

if multiple_img == False:
    results = model.predict(source=image_file, iou=0.5, agnostic_nms=True)
    boxes = results[0].boxes 
    pred = []
    for b in boxes:
        if torch.cuda.is_available():
            xyxyn = b.xyxyn[0]
            pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
    predictions = torch.tensor(pred)
    img = cv.imread(image_file)
    save_image_predictions(predictions, image_file, save_dir, class_colours, classes)
else:
    imglist = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
    imgsave_dir = os.path.join(save_dir, 'detections', 'detections_images')
    os.makedirs(imgsave_dir, exist_ok=True)
    for i, imgname in enumerate(imglist):
        print(f'predictions on {i+1}/{len(imglist)}')
        if i >= 1000: # for debugging purposes
            break
        results = model.predict(source=imgname, iou=0.5, agnostic_nms=True)
        boxes = results[0].boxes 
        pred = []
        for b in boxes:
            if torch.cuda.is_available():
                xyxyn = b.xyxyn[0]
                pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
        predictions = torch.tensor(pred)
        save_image_predictions(predictions, imgname, imgsave_dir, class_colours, classes)


# for interactive debugger in terminal:
import code
code.interact(local=dict(globals(), **locals()))