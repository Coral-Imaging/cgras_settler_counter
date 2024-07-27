#! /usr/bin/env python3

"""resize_files.py

resize data files in sepcified folder to specified resolutions for training at different resolutions
Dorian Tsai
2024 July 26
"""
  
import os
import cv2 as cv
import glob
  
# specify data directory
# should be tiled images

data_dir = '/Users/doriantsai/Code/cgras_settler_counter/resolution_experiment/images'
print(f'data_dir = {data_dir}')

# specify resolution(s)/desired image sizes
image_sizes = [480, 320, 160]
print(f'image_sizes = {image_sizes}')

# loop over image sizes, resize and save into folder
for i in image_sizes:
    
    image_list = glob.glob(os.path.join(data_dir,'*.jpg'))
    print(f'length of image_list = {len(image_list)}')
    
    # get aspect ratio, assume all images in same dir have same aspect ratio
    if len(image_list) > 1:
        image = cv.imread(image_list[0])
        height, width, chan = image.shape
        ar = width / height
    else:
        error('no images in image_list')
    
    # specify output data directory
    out_dir = os.path.join(data_dir,str(i)+'p')
    os.makedirs(out_dir, exist_ok=True)
    print(f'out_dir = {out_dir}')
    
    # loop over all images in list
    for image_name in image_list:
        
        # open image (bgr)
        image = cv.imread(image_name)
        print(f'reading image = {image_name}')
        
        # resize image, assuming width is largest dimension
        width_r = i
        height_r = round(width_r / ar)
        image_r = cv.resize(image, (width_r, height_r), interpolation=cv.INTER_LINEAR)
        
        # save image (bgr)
        save_file = os.path.join(out_dir, os.path.basename(image_name)[:-4]+'_'+str(i)+'.jpg')
        print(f'saving image = {save_file}')
        cv.imwrite(save_file, image_r)
    
print('done')
import code
code.interact(local=dict(globals(), **locals()))