#! /usr/bin/env python3

""" cvatcoco_to_yolo.py
    This script is used to convert a COCO dataset to YOLO format.
"""

from ultralytics.data.converter import convert_coco
import os
import zipfile
import shutil
import glob
import random

##TODO Fix code to make sure stuff doesn't get miss labeled in converstion
# currently coco top list of classes, does not match order of classes in yolo or cvat
# fix would be reading all labels and changing, or updating yaml file to match order of classes in coco (and utils and code list of classes)

#export from CVAT as COCO1.1
#comes in as ziped file with annotations/instances_default.json
file_name = 'cgras_iteration2_coco.zip' #'cgras_coco.zip'
save_dir = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/cvat_labels'
download_dir = '/home/java/Downloads'
#data_locaton = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/cgras_data_copied_2240605_ultralytics_data' #location of /images folder
data_locaton = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/tilling2/'
convert = False
label_fix = False #true if coco has different order of classes
fill_in = False #if want to fill in blank text files

if convert:
    # convert coco to yolo
    with zipfile.ZipFile(os.path.join(download_dir, file_name), 'r') as zip_ref:
        zip_ref.extractall(download_dir)
            #annotation folder in downloads

    downloaded_labels_dir = os.path.join(download_dir, 'annotations')
    lable_dir = os.path.join(save_dir, 'labels')
    coco_labels = os.path.join(lable_dir, 'default')

    convert_coco(labels_dir=downloaded_labels_dir, save_dir=save_dir,
                    use_segments=True, use_keypoints=False, cls91to80=False)

    #move from subfolder in coco/labels/defult to where data is
    data_labels = os.path.join(data_locaton, 'labels')
    os.makedirs(data_labels, exist_ok=True)
    for filename in os.listdir(coco_labels):
        source = os.path.join(coco_labels, filename)
        destination = os.path.join(data_labels, filename)
        if os.path.isfile(source):
            shutil.move(source, destination)
    shutil.rmtree(save_dir) #tidy up

    print("conversion complete")
    import code
    code.interact(local=dict(globals(), **locals()))

if label_fix:
    for i, label in enumerate(glob.glob(os.path.join(data_locaton, 'labels', '*.txt'))):
        print("processing label: ", label)
        if i<=2:
            continue
        with open(label, 'r') as file:
            data = file.readlines()
        with open(label, 'w') as file:
            for line in data:
                line = line.split()
                if line[0] == '3':
                    line[0] = '4'
                elif line[0] == '4':
                    line[0] = '5'
                elif line[0] == '5':
                    line[0] = '6'
                elif line[0] == '6':
                    line[0] = '7'
                elif line[0] == '7':
                    line[0] = '8'
                elif line[0] == '8':
                    line[0] = '9'
                elif line[0] == '9':
                    line[0] = '10'
                elif line[0] == '10':
                    line[0] = '3'
                line[0] = str(int(line[0]))
                file.write(' '.join(line)+'\n')
    print("label fix complete")
    import code
    code.interact(local=dict(globals(), **locals()))

#if images with no annotations
#need to add in a bank text file for labels
if fill_in:
    images_folder = os.path.join(save_dir,'images')
    labels_folder = os.path.join(save_dir,'labels')
    image_files = [os.path.splitext(file)[0] for file in os.listdir(images_folder)]
    label_files = [os.path.splitext(file)[0] for file in os.listdir(labels_folder)]
    missing_files = [file for file in image_files if file not in label_files]
    print(f'number of missing files: {len(missing_files)}')
    for file_name in missing_files:
        with open(os.path.join(labels_folder, file_name+'.txt'), 'w') as file:
            pass
    print ("blank text files added")


import code
code.interact(local=dict(globals(), **locals()))