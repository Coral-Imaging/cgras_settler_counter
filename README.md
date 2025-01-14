# CGRAS Settler Counter

**Coral Growout Robotic Assessment System (CGRAS) Settler Counter**  
This repository contains code to count coral settlers in settlement tanks using object detection and/or image segmentation (TBD).

Last updated: Jan 2024
---

## Code Layout

### Annotation
- **`CVAT_class_constructor.json`**: Defines CVAT classes and colors for CGRAS.  
- **`cvat1.1_to_yolo.py`**: Converts a CVAT dataset to YOLO format.  
- **`cvatcoco_to_yolo.py`**: Converts a COCO dataset to YOLO format.  
- **`predict_boxes.py`**: Saves predicted bounding box results as `.txt` and `.jpg`.  
- **`predict_to_cvat.py`**: Runs a trained YOLOv8 segment model on unlabeled images, saving results in CVAT annotation format (for human-in-the-loop processing).  
- **`relabel.py`**: Script to fix or change class labels of a dataset.  
- **`roboflow_sahi.py`**: Implements SAHI for image processing, useful for visualization or human-in-the-loop tasks via CVAT.  
- **`run_predict.sh`**: Bash script to batch process the `predict_to_cvat` script.  
- **`Utils.py`**: Contains class definitions and utility functions for other scripts.  
- **`view_predictions.py`**: Visualizes prediction results using a trained YOLOv8 weights file.

---

### Classifier
- **`relabel_seg_to_single_class.py`**: Converts YOLO segmentation labels with multiple classes into a single class.  
- **`segTOclassifier.py`**: Processes YOLO segmentation data to generate cropped images for specific class instances.

---

### Image Processing
Functions and scripts for processing images:  
- **`extract_imgs.py`**: Extracts images from CGRAS data samples into a structured folder format for detection.  
- **`splitfiles.py`**: Splits a dataset into training, testing, and validation sets.  
- **`tiling_images.py`**: Tiles large images into smaller ones with annotations for YOLO model training.  

#### ROI and ImageJ Tools
- **`convert_tif2jpg.py`**: Converts TIFF files to JPG format.  
- **`pde_to_cvat.py`**: Converts ROI PDE data to CVAT format.  
- **`ROI_2_CSV.py`**: Converts ROI PDE data to CSV format.  
- **`run_ROI_2_CSV.py`**: Bash script for batch processing ROI to CSV conversion.

---

### Resolution Experiment
- **`remove_too_many_negs.py`**: Removes negative (unlabeled) images from a dataset.  
- **`resize-files.py`**: Resizes data files to specified resolutions for training experiments.  
- **`resolution_script.py`**: Script for testing the impact of different resolutions on model performance.

---

### Segmenter
Scripts for training and predicting coral settlers using a YOLO model:  
- **`cgras_20230421.yaml`**: YAML file specifying training data locations.  
- **`cgras_hpc.yaml`**: YAML file for HPC data paths.  
- **`cgras_Pde_20230421.yaml`**: YAML file specifying PDE data locations.  
- **`Pdae_train_segmenter.py`**: Code for training a YOLO model using PDE data.  
- **`predict_segmenter.py`**: Code for predicting corals in images.  
- **`train_segmenter.py`**: Script to train a YOLO model on a dataset.  
- **`val_segmenter.py`**: Evaluates test data to measure model performance.

---

### Legacy Code
- **`Annotations.py`**: Tests SAM annotation.  
- **`poly_to_mask.py`**: Converts polygons to masks for CVAT annotation.  
- **`till_n_predict.py`**: Tiles large images and visualizes model predictions.  
- **`min_res.py`**: Runs experiments to determine the minimum resolution for coral detection.  
- **`segment_cgras_images.py`**: Tests SAM annotations.  
- **`original_sahi_yolov8`, `updated_sahi_yolov8`**: Updates to YOLOv8 SAHI model, later superseded by `roboflow_sahi`.  
- **`temp_calc.py`**: Uses confusion matrices to compute YOLO model performance metrics.

---  

## Contact Java
Java Terry can be reached at '33javalava@gmail.com' or on '0444 584 863'
