#! /usr/bin/env python3

""" convert_tiff2jpg.py
"""

from PIL import Image
import os

# tiff directory
tif_dir = '/home/dorian/Data/cgras_dataset_20230421/tif'


# jpg out directory
jpg_dir = '/home/dorian/Data/cgras_dataset_20230421/jpg'
os.makedirs(jpg_dir, exist_ok=True)

for i, filename in enumerate(os.listdir(tif_dir)):
    if filename.endswith('.tif'):
        print(f'converting {i}: {filename}')
        # open TIFF image
        image = Image.open(os.path.join(tif_dir, filename))
        # convert TIFF to jpeg
        new_filename = filename.rsplit('.', 1)[0] + '.jpg'
        image.convert('RGB').save(os.path.join(jpg_dir, new_filename))

        image.close()

print('Conversion complete')
