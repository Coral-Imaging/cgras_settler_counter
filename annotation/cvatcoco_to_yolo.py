from ultralytics.data.converter import convert_coco
import os
import zipfile
import shutil

#export from CVAT as COCO1.1
#comes in as ziped file with annotations/instances_default.json
file_name = 'coco_5_images_polygons.zip'
save_dir = '/home/java/Java/data/new_data'
download_dir = '/home/java/Downloads'

with zipfile.ZipFile(os.path.join('/home/java/Downloads', file_name), 'r') as zip_ref:
   zip_ref.extractall('/home/java/Downloads')
    #annotation folder in downloads

downloaded_labels_dir = os.path.join(download_dir, 'annotations')
lable_dir = os.path.join(save_dir, 'labels')
coco_labels = os.path.join(lable_dir, 'default')

convert_coco(labels_dir=downloaded_labels_dir, save_dir=save_dir,
                 use_segments=True, use_keypoints=False, cls91to80=False)

for filename in os.listdir(coco_labels):
    source = os.path.join(coco_labels, filename)
    destination = os.path.join(lable_dir, filename)
    if os.path.isfile(source):
        shutil.move(source, destination)
