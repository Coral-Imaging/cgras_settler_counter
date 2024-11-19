#! /usr/bin/env python3

"""trian_segmenter.py
train basic yolov8 model for image segmentation
"""

from ultralytics import YOLO
import torch

data_file = '/home/wardlewo/Reggie/corals/cgras_settler_counter/segmenter/cgras_20230421.yaml'
#data_file = sys.argv[1]

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO('yolov8x-seg.pt').to(device)
             #/home/java/Java/ultralytics/runs/segment/train9/weights/cgras_yolov8n-seg_640p_20231209.pt').to(device)

# train model
# for base training only want, 0: 'recruit_live_white', 1: 'recruit_cluster_live_white', 
# 6: 'recruit_dead', 7: 'recruit_cluster_dead', 9: 'pest_tubeworm'. 10: 'unknown'

#model.train(data=data_file, epochs=200, batch=10)

# classes arg is lightweight and simply ignore classes that are not included in the classes list, 
# during train, val and predict, it has no effect on model architecture.
model.train(data=data_file, epochs=1, batch=1,  classes=[0,1,2,3,6,7,8,9]) #test run
#model.train(data=data_file, epochs=100, batch=-1, classes=[0,1,6,7,9,10], imgsz=1280, nbs=12)

# print('Model Inference:')
# image_file = '/home/java/Java/data/cgras_20230421/train/images/00_20230116_MIS1_RC_Aspat_T04_08.jpg'
# results = model.predict(source=image_file)
# print(results)

# # for interactive debugger in terminal:
# import code
# code.interact(local=dict(globals(), **locals()))