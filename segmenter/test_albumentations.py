#!/usr/bin/env python3

# test albumentations, because having error with randomResizedCrop

import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

def augment_img(aug, image):
    image_array = np.array(image)
    augmented_img = aug(image=image_array)['image']
    return augmented_img

# image = np.ones((300, 300, 3)).astype(np.float32)
image = ski.data.camera()
print(type(image))

print('test albumentations')

transform = A.RandomResizedCrop(height=20,width=120, scale=(0.4, 0.5), p=1)

b = augment_img(transform, image)

plt.figure()
plt.imshow(image)

plt.figure()
plt.imshow(b)
plt.show()