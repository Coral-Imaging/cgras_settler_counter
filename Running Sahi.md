# Running Sahi instuctions

The following instructions are for using [Sahi](https://github.com/obss/sahi) while detecting corals.
Is should be noted the currently Sahi and yolov8 only support object detection and not object segmentation.

## 1. 
Run the script test_sahi.py (saved in the annotations folder) updating the export directory, image file list and model path as required (all defined around lines 95)
The code will preduce a image with detected corals in a bounding box with a label of class name and confidence of the detection for every image specified in the image file list.

## Other notes about the script
Code commented out above line 95 is for some basic file manipulation and should be left commented out unless there needs be files moved.
The save_image_predictions_bb function is the exact same as the function used in predict_boxes.py
Code commented below line 190 if a rough draft of the code for completing the human in the loop pipline, it should remain commented out.