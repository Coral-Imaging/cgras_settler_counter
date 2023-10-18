from ultralytics.data.converter import convert_coco

convert_coco(labels_dir='/home/java/Downloads/cgras_20231810_polygon_coco/annotations/',
                 use_segments=True, use_keypoints=False, cls91to80=False)

# will save txt files in source dir of where python script is called + '/yolo_labels/labels/default/'
# ie. home/java/Java/cgras/cgras_settler_counter/yolo_labels/labels/default