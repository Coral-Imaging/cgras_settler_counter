#! /usr/bin/env python3

"""trian_segmenter.py
train basic yolov8 model for image segmentation
"""

from ultralytics import YOLO
import os
import glob
import torch

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO('home/java/Java/cgras/cgras_settler_counter/yolov8x-seg.pt')

# train model
data_file = '/home/java/Java/cgras/cgras_settler_counter/segmenter/cgras_20230421.yaml'
#model.train(data=data_file, epochs=200, batch=10)
model.train(data=data_file, epochs=10, batch=10)

print('Model Inference:')
image_file = '/home/java/Java/data/cgras_dataset_20230421/train/images/00_20230116_MIS1_RC_Aspat_T04_08.jpg'
results = model(image_file)
print(results)



# for interactive debugger in terminal:
import code
code.interact(local=dict(globals(), **locals()))