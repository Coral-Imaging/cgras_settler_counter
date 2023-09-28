"""Testing converstion of polygons to masks for CVAT annotation.
    Download polygon annotations using CVAT1.1 format
    Upload mask annotations using CVAT1.1 format after zipping the outputed XML file
"""
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree
from itertools import groupby
import code
import matplotlib.pyplot as plt

def binary_mask_to_rle(binary_mask):
    """binary_mask_to_rle
    Convert a binary np array into a RLE format
    NOTE: I think this is the function with an error in it. - Java

    Args:
        binary_mask (uint8 2D numpy array): binary mask

    Returns:
        rle: list of rle numbers
    """
    rle = []
    for i, (amount, number) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and amount == 1: #first rle is 0, so if 1 is first, add single 0
           rle.append(0)
        rle.append(len(list(number)))
    return rle

def poly_2_rle(points):
    """poly_2_rle
    Converts a set of points for a polygon into an rle string and saves data

    Args:
        points (2D numpy array)
    
    Returns:
        rle_string: string of the rle numbers,
        left: (int) left positioning of the rle numbers in pixles,
        top: (int) top positioning of the rle numbers in pixles,
        width: (int) width of the bounding box of the rle numbers in pixels,
        height: (int) height of the bounding box of the rle numbers in pixles
    """
    # create bounding box
    left = int(np.min(points[:, 0]))
    top = int(np.min(points[:, 1]))
    right = int(np.max(points[:, 0]))
    bottom = int(np.max(points[:, 1]))
    width = right - left + 1
    height = bottom - top + 1

    # create mask size of bounding box
    mask = np.zeros((height, width), dtype=np.uint8)
    # points relative to bounding box
    points[:, 0] -= left
    points[:, 1] -= top
    # fill mask where points are
    cv2.fillPoly(mask, [points.astype(int)], color=1)

    #code.interact(local=dict(globals(), **locals()))
    # visual check of mask - looks good
    SHOW_IMAGE = False
    if (SHOW_IMAGE):
        plt.imshow(mask, cmap='binary')
        plt.show()

    mask_rle = binary_mask_to_rle(mask)

    # rle string
    rle_string = ",".join(map(str, mask_rle))
    
    return rle_string, left, top, width, height

# load XML file with the polygons
source_file = 'polyshapes.xml'
output_filename = 'output_xml_file.xml'
tree = ET.parse(source_file)
root = tree.getroot() 

# Create a new XML ElementTree
new_tree = ElementTree(Element("annotations"))

# add version element
version_element = ET.Element('version')
version_element.text = '1.1'
new_tree.getroot().append(version_element)

# add Meta elements, (copy over from source_file)
meta_element = root.find('.//meta')
if meta_element is not None:
    new_meta_elem = ET.SubElement(new_tree.getroot(), 'meta')

    # copy all subelements of meta
    for sub_element in meta_element:
        new_meta_elem.append(sub_element)

# iterate through image elements
for image_element in root.findall('.//image'):
    image_id = image_element.get('id')
    image_name = image_element.get('name')
    image_width = int(image_element.get('width'))
    image_height = int(image_element.get('height'))

    # create new image element in new XML
    new_elem = SubElement(new_tree.getroot(), 'image')
    new_elem.set('id', image_id)
    new_elem.set('name', image_name)
    new_elem.set('width', str(image_width))
    new_elem.set('height', str(image_height))

    # acess polygon annotations in the image
    for polygon_element in image_element.findall('polygon'):
        polygon_points = polygon_element.get('points')
        polygon_label = polygon_element.get('label')
        polygon_z_order = polygon_element.get('z_order')

        # points into np array 
        points = np.array([list(map(float, point.split(',')))
                          for point in polygon_points.split(';')])

        # convert the points into rle format
        rle_string, left, top, width, height = poly_2_rle(points)

        #XML mask element details
        mask_elem = SubElement(new_elem, 'mask')
        mask_elem.set('label', polygon_label)
        mask_elem.set('source', 'semi-auto')
        mask_elem.set('occluded', '0')
        mask_elem.set('rle', rle_string)
        mask_elem.set('left', str(left))
        mask_elem.set('top', str(top))
        mask_elem.set('width', str(width))
        mask_elem.set('height', str(height))
        mask_elem.set('z_order', polygon_z_order)

# Write modified XML to output file
new_tree.write(output_filename, encoding='utf-8', xml_declaration=True)

code.interact(local=dict(globals(), **locals()))