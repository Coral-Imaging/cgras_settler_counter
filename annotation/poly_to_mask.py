"""Converstions of polygons to masks for CVAT annotation.
    Download polygon annotations using CVAT1.1 format
    Upload mask annotations using CVAT1.1 format after zipping the outputed XML file
"""
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree
import code
import matplotlib.pyplot as plt
import zipfile
from Utils import binary_mask_to_rle, poly_2_rle

def rle_to_binary_mask(rle_list, 
                       width: int, 
                       height: int, 
                       SHOW_IMAGE: bool):
    """rle_to_binary_mask
    Converts a rle_list into a binary np array. Used to check the binary_mask_to_rle function

    Args:
        rle_list (list of strings): containing the rle information
        width (int): width of shape
        height (int): height of shape
        SHOW_IMAGE (bool): True if binary mask wants to be viewed

    Returns:
        mask: uint8 2D np array
    """
    mask = np.zeros((height, width), dtype=np.uint8) 
    current_pixel = 0
    
    for i in range(0, len(rle_list)):
        run_length = int(rle_list[i]) #find the length of current 0 or 1 run
        if (i % 2 == 0): #if an even number the pixel value will be 0
            run_value = 0
        else:
            run_value = 1

        for j in range(run_length): #fill the pixel with the correct value
            mask.flat[current_pixel] = run_value 
            current_pixel += 1

    if (SHOW_IMAGE):
        print("rle_list to binary mask")
        plt.imshow(mask, cmap='binary')
        plt.show()

    return mask

def maskxml_to_polyxml(source_file: str,
                       output_filename: str):
    '''NOT TESTED!!!   maskxml_to_polyxml
    Converts a cvat1.1 source file annotation with masks to an cvat1.1 outputfile with annontations as polygons.
    NOTE: '.xml' must be included in both source_file and output_filename

    Args:
        source_file (str): name of source file with annontations
        output_filename (str): name of output file to save to.
    '''
    # load XML file with the polygons via a tree
    tree = ET.parse(source_file)
    root = tree.getroot() 
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
    i = 0
    for image_element in root.findall('.//image'):
        i += 1
        print(i,'images being processed')
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

        # acess mask annotations in the image
        for mask_ele in image_element.findall('mask'):
            mask_rle = mask_ele.get('rle')
            mask_width = mask_ele.get('width')
            mask_height = mask_ele.get('height')
            # rle into list 
            rle_list = list(map(int, mask_rle.split(',')))
            # convert the rle into mask
            mask = rle_to_binary_mask(rle_list, int(mask_width), int(mask_height), SHOW_IMAGE=False)
            # convert the mask into polygon
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # convert the polygon into points
            points = np.squeeze(contours[0])
            # convert the points into string
            points_string = ';'.join(','.join(str(point) for point in points))
            #XML polygon element details
            poly_elem = SubElement(new_elem, 'polygon')
            poly_elem.set('label', mask_ele.get('label'))
            poly_elem.set('points', points_string)
            poly_elem.set('z_order', mask_ele.get('z_order'))
        print(len(image_element.findall('mask')),'masks converted into polygons',image_name)
        
    new_tree.write(output_filename, encoding='utf-8', xml_declaration=True)

    #Zip the file
    zip_filename = output_filename.split('.')[0] + '.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_filename, arcname='output_xml_file.xml')
    print('XML file zipped')


def polyxml_to_maskxml(source_file: str, 
                       output_filename: str, 
                       SHOW_IMAGE: bool):
    '''polyxml_to_maskxml
    Converts a cvat1.1 source file annotation with polyshapes to an cvat1.1 outputfile with annontations as masks.
    NOTE: '.xml' must be included in both source_file and output_filename

    Args:
        source_file (str): name of source file with annontations
        output_filename (str): name of output file to save to.
        SHOW_IMAGE (bool): True if polygons and rle want to be viewed (to check process)
    '''
# load XML file with the polygons via a tree
    tree = ET.parse(source_file)
    root = tree.getroot() 
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
    i = 0
    for image_element in root.findall('.//image'):
        i += 1
        print(i,'images being processed')

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
            rle_string, left, top, width, height = poly_2_rle(points, ',', SHOW_IMAGE)

             # If converstion needs to be checked
            rle_list = list(map(str, rle_string.split(','))) #rle into list
            test_mask = rle_to_binary_mask(rle_list, width, height, SHOW_IMAGE)  # convert the rle into mask
            
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
        print(len(image_element.findall('polygon')),'polgons converted in image',image_name)
        #copy over any mask elements
        for o_mask_ele in image_element.findall('mask'):
            c_mask_elem = SubElement(new_elem, 'mask')
            c_mask_elem.set('label', o_mask_ele.get('label'))
            c_mask_elem.set('source', 'semi-auto')
            c_mask_elem.set('occluded', '0')
            c_mask_elem.set('rle', o_mask_ele.get('rle'))
            c_mask_elem.set('left', o_mask_ele.get('left'))
            c_mask_elem.set('top', o_mask_ele.get('top'))
            c_mask_elem.set('width', o_mask_ele.get('width'))
            c_mask_elem.set('height', o_mask_ele.get('height'))
            c_mask_elem.set('z_order', o_mask_ele.get('z_order'))
        print(len(image_element.findall('mask')),'masks kept in image',image_name)
        

    # Write modified XML to output file
    new_tree.write(output_filename, encoding='utf-8', xml_declaration=True)

    #Zip the file
    zip_filename = output_filename.split('.')[0] + '.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_filename, arcname='output_xml_file.xml')
    print('XML file zipped')

    code.interact(local=dict(globals(), **locals()))

def test_rle_to_mask(source_file: str):
    '''test_rle_to_mask
    Function to check that the rle_to_binary_mask function works

    Args:
        source_file (str): name of sourcefile
    '''
    # load XML file with the mask
    tree = ET.parse(source_file)
    root = tree.getroot() 

    for image_element in root.findall('.//image'):
        for mask_element in image_element.findall('mask'):
            mask_rle = mask_element.get('rle')
            mask_width = mask_element.get('width')
            mask_height = mask_element.get('height')
            # rle into list 
            rle_list = list(map(int, mask_rle.split(',')))
            # convert the rle into mask
            mask = rle_to_binary_mask(rle_list, int(mask_width), int(mask_height), SHOW_IMAGE=True)

#test_rle_to_mask(source_file='masks.xml')

def main():
   polyxml_to_maskxml(source_file='/home/java/Downloads/cgras_20231028/annotations.xml', output_filename='cgras_20231028_masks.xml', SHOW_IMAGE=False)

if __name__ == "__main__":
    main()
