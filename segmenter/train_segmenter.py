#! /usr/bin/env python3

"""trian_segmenter.py
train basic yolov8 model for image segmentation
"""

from ultralytics import YOLO
import os
import glob


# load model
model = YOLO('yolov8x-seg.pt')

# train model
data_file = 'cgras_20230421.yaml'
model.train(data=data_file, epochs=200, batch=10)


print('Model Inference:')
image_file = '/home/dorian/Data/cgras_datasets/cgras_dataset_20230421/train/images/00_20230116_MIS1_RC_Aspat_T04_08.jpg'
results = model(image_file)
print(results)



# for interactive debugger in terminal:
import code
code.interact(local=dict(globals(), **locals()))