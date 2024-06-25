# cgras_settler_counter

Coral Growout Robotic Assessment System (CGRAS) Settler Counter

Code to count coral settlers in settlement tanks using object detection and/or image segmentation (TBD).

## Code layout
### Segmenter
Contains code to train or predict coral using a yolo model
cgras_20230421.yml - file that stores the location of data to train on
predict_segmenter.py - code to predict corals on a list of images
train_segmenter.py - code to train a yolo model on a datset
### Image Processing
Functions or code to process images
convert_tif2jpg.py - converts a list of tiffs to jpg
extract_imgs.py - extract images from cgras data samples into folder structure for detection
tiling_images.py - tile big images into smaller ones with annotations that can then be trained on via a yolo model
## Annotation
Annotations.py - annotation class to segment and create CVAT annotations
CVAT_class_constructor.json - CVAT classes and colours for CGRAS
poly_to_mask.py - Converstions of polygons to masks for CVAT annotation.
predict_boxes.py - save predicted bounding box results both as txt and .jpg.
predict_to_cvat.py - script to run a trained yolov8 segment model on unlabeled images, saving these results in cvat annotation form
roboflow_sahi.py - SAHI over images, can be used to visualise predictions or for humman in the the loop processing via cvat (see Human_in_the_loop.)
Utils.py - class definitions and functions for use in other scripts
view_predictions.py - visualise the predition results using a trained yolov8 weights file (ideally from a trained run)

Old/Superceeded scripts
 original_sahi_yolov8, updated_sahi_yolov8 - updates to the yolov8 sahi model to try and get sahi ultralitics working. Supperceeded by roboflow_sahi
 test_sahi - using sahi ultralitics 
 