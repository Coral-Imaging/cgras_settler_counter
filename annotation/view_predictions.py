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
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree

# save_dir = '/home/java/Java/ultralytics/runs/segment/train3'
weights_file_path = '/home/java/Java/ultralytics/runs/segment/train4/weights/best.pt'
image_file = '/home/java/Java/data/cgras_20230421/train/images/00_20230116_MIS1_RC_Aspat_T04_08.jpg'
# img_folder ='some/path/to/a/bunch/of/images'
# multiple_img = False
save_dir = '/home/java/Java/data/cgras_20231028'
#img_folder = '/home/java/Java/data/cgras_20231028/images'
img_folder = '/home/java/Java/data/cgras_20230421/train/images'
multiple_img = False
plot_masks = True

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


def save_image_predictions_mask(results, image, imgname, save_path, class_list, classes, class_colours):
    m_idx_list = []
    height, width, _ = image.shape
    masked = image.copy()
    for m in results[0].masks:
        mask_raw = m.cpu().data.numpy().transpose(1, 2, 0)
        mask_3channel = cv.merge((mask_raw,mask_raw,mask_raw)) # Convert single channel grayscale to 3 channel image
        mask = cv.resize(mask_3channel, (width, height)) # Resize the mask to the same size as the image
        mask_indices = np.where(mask != 0)
        m_idx_list.append(mask_indices)

    for i, m in enumerate(m_idx_list):
        desired_color = class_colours[classes[int(class_list[i])]]
        masked[m[0], m[1]] = desired_color
    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det.jpg')
    cv.imwrite(imgsave_path, semi_transparent_mask)
    print(imgsave_path)

    import code
    code.interact(local=dict(globals(), **locals()))

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_file_path).to(device)

# get predictions
print('Model Inference:')

if 1 ==2: #ultralytics code
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
        im_array = r.plot(conf=True, line_width=4, font_size=4, boxes=False)
        im = Image.fromarray(im_array[..., ::-1])
        im.show()
    #r.save_txt(image_file[:-4]+'_'+ '_detmask.txt', save_conf=True ) 
        ### Saves masks like: (class being first number and confidence being the last, middle numbers the mask.xyn values ) masks[j].xyn[0].copy().reshape(-1)
        ### 9 0.207813 0.368893 0.20625 0.371234 0.20625 0.392305 0.207813 0.394646 0.215625 0.394646 0.217187 0.392305 0.217187 0.389963 0.220313 0.385281 0.221875 0.385281 0.223438 0.38294 0.223438 0.371234 0.221875 0.368893 0.971601
    import code
    code.interact(local=dict(globals(), **locals()))
if 1==1: #multiple images
    imglist = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
    imgsave_dir = os.path.join(save_dir, 'detections', 'detections_txt')
    os.makedirs(imgsave_dir, exist_ok=True)
    for i, imgname in enumerate(imglist):
        print(f'predictions on {i+1}/{len(imglist)}')
        if i >= 10: # for debugging purposes
            break
        image = cv.imread(imgname)
        results = model.predict(source=imgname, iou=0.5, agnostic_nms=True)
        conf, class_list = [], [], 
        for i, b in enumerate(results[0].boxes):
            conf.append(b.conf.item())
            class_list.append(b.cls.item())
        #save_image_predictions_mask(results, image, imgname, imgsave_dir, class_list, classes, class_colours)
        imgsavename = os.path.basename(imgname)
        #save_path = os.path.join(imgsave_dir, imgsavename[:-4] + '.txt')
        #save_txt_predictions_masks(results, conf, class_list, save_path)
        save_image_predictions_mask(results, image, imgname, save_dir, class_list, classes, class_colours)

if 1==2: #single image, all masks
    image = cv.imread(image_file)
    results = model.predict(source=image_file, iou=0.5, agnostic_nms=True)
    
    imgsave_path = image_file[:-4]+'_detmask.jpg'
    save_image_predictions_mask(results, image, imgsave_path, class_list, classes, class_colours)
    import code
    code.interact(local=dict(globals(), **locals()))
    conf, class_list = [], [], 
    for i, b in enumerate(results[0].boxes):
        conf.append(b.conf.item())
        class_list.append(b.cls.item())

    save_path = image_file[:-4]+'_detmask.txt'
    save_txt_predictions_masks(results, image, image_file, save_path, class_list, classes, class_colours)

if 2==3: #using boxes, plot boxes (works)
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


def binary_mask_to_rle(binary_mask):
    """binary_mask_to_rle
    Convert a binary np array into a RLE format
    
    Args:
        binary_mask (uint8 2D numpy array): binary mask

    Returns:
        rle: list of rle numbers
    """
    rle = []
    current_pixel = 0
    run_length = 0

    for pixel in binary_mask.ravel(order='C'): #go through the flaterened binary mask
        if pixel == current_pixel:  #increase number if same pixel
            run_length += 1
        else: #else save run length and reset
            rle.append(run_length) 
            run_length = 1
            current_pixel = pixel
    return rle

def poly_2_rle(points, 
               SHOW_IMAGE: bool):
    """poly_2_rle
    Converts a set of points for a polygon into an rle string and saves the data

    Args:
        points (2D numpy array): points of polygon
        SHOW_IMAGE (bool): True if binary mask wants to be viewed
    
    Returns:
        rle_string: string of the rle numbers,
        left: (int) left positioning of the rle numbers in pixles,
        top: (int) top positioning of the rle numbers in pixles,
        width: (int) width of the bounding box of the rle numbers in pixels,
        height: (int) height of the bounding box of the rle numbers in pixles
    """
    # create bounding box
    left = int(np.min(points[:, 0]))
    top = int(np.min(points[:, 1]))
    right = int(np.max(points[:, 0]))
    bottom = int(np.max(points[:, 1]))
    width = right - left + 1
    height = bottom - top + 1

    # create mask size of bounding box
    mask = np.zeros((height, width), dtype=np.uint8)
    # points relative to bounding box
    points[:, 0] -= left
    points[:, 1] -= top
    # fill mask where points are
    cv.fillPoly(mask, [points.astype(int)], color=1)

    # visual check of mask - looks good
    #SHOW_IMAGE = False
    if (SHOW_IMAGE):
        plt.imshow(mask, cmap='binary')
        plt.show()

    # convert the mask into a rle
    mask_rle = binary_mask_to_rle(mask)

    # rle string
    rle_string = ",".join(map(str, mask_rle))
    
    return rle_string, left, top, width, height

base_file = "/home/java/Downloads/annotations.xml"
base_img_location = "/home/java/Java/data/cgras_20230421/train/images"
output_filename = "/home/java/Downloads/complete.xml"

tree = ET.parse(base_file)
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

# iterate through image elements
i = 0
for image_element in root.findall('.//image'):
    i += 1
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
    
    image_file = os.path.join(base_img_location,image_name)
    results = model.predict(source=image_file, iou=0.5, agnostic_nms=True)
    masks = results[0].masks
    class_list = [b.cls.item() for b in results[0].boxes]

    for j, m in enumerate(masks):
        label = classes[int(class_list[j])]
        mxy = m.xy
        xy = np.squeeze(mxy)
        try:
            rle_string, left, top, width, height  = poly_2_rle(xy,False)
            mask_elem = SubElement(new_elem, 'mask')
            mask_elem.set('label', label)
            mask_elem.set('source', 'semi-auto')
            mask_elem.set('occluded', '0')
            mask_elem.set('rle', rle_string)
            mask_elem.set('left', str(left))
            mask_elem.set('top', str(top))
            mask_elem.set('width', str(width))
            mask_elem.set('height', str(height))
            mask_elem.set('z_order', '0')
        except:
            print(f'mask {j} encountered problem xy = {xy}')

    print(len(class_list),'masks converted in image',image_name)

new_tree.write(output_filename, encoding='utf-8', xml_declaration=True)



# for interactive debugger in terminal:
import code
code.interact(local=dict(globals(), **locals()))