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
from annotation.Utils import binary_mask_to_rle, poly_2_rle

def validate_rle(rle_list, width, height):
    total_pixels = width * height
    pixel_count = sum(rle_list)
    
    if pixel_count != total_pixels:
        print(f"Warning: RLE data does not match the total number of pixels. RLE pixels: {pixel_count}, Expected pixels: {total_pixels}")
        return False
    
    return True

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
    total_pixels = width * height

    # if not validate_rle(rle_list, width, height):
    #     import code
    #     code.interact(local=dict(globals(), **locals()))

    for i in range(0, len(rle_list)):
        run_length = int(rle_list[i]) #find the length of current 0 or 1 run
        if (i % 2 == 0): #if an even number the pixel value will be 0
            run_value = 0
        else:
            run_value = 1
        for j in range(run_length): # fill the pixel with the correct value
            mask.flat[current_pixel] = run_value 
            current_pixel += 1

    if (SHOW_IMAGE):
        print("rle_list to binary mask")
        plt.imshow(mask, cmap='binary')
        plt.show()

    return mask

def maskxml_to_polyxml(source_file: str,
                       output_filename: str):
    '''maskxml_to_polyxml
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
            mask_width = int(mask_ele.get('width'))
            mask_height = int(mask_ele.get('height'))
            mask_top = int(mask_ele.get('top'))
            mask_left = int(mask_ele.get('left'))
            # rle into list 
            rle_list = list(map(int, mask_rle.split(',')))
            # convert the rle into mask
            try:
                mask = rle_to_binary_mask(rle_list, mask_width, mask_height, SHOW_IMAGE=False)
            except:
                print("error in rle_to_binary_mask")
                import code
                code.interact(local=dict(globals(), **locals()))
            # convert the mask into polygon and get in right format
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 3:
                    print(f"Contour with insufficient points found for image {image_name}: {len(contour)} points")
                    continue
                points = np.squeeze(contour)
                if len(points.shape) != 2 or points.shape[1] != 2:
                    print(f"Unexpected points shape for image {image_name}: {points.shape}")
                    continue

                formatted_points = ';'.join([f"{x+mask_left},{y+mask_top}" for x, y in points])
                # XML polygon element details
                poly_elem = SubElement(new_elem, 'polygon')
                poly_elem.set('label', mask_ele.get('label'))
                poly_elem.set('points', formatted_points)
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
   #polyxml_to_maskxml(source_file='/home/java/Downloads/cgras_20231028/annotations.xml', output_filename='cgras_20231028_masks.xml', SHOW_IMAGE=False)
   #TODO Add auto extract from file
    maskxml_to_polyxml(source_file='/home/java/Downloads/cgras_iteration2/cgras_iteration2.xml', output_filename='/home/java/Downloads/cgras_iteration2/polygons.xml')

def calculate_area(points):
    x_coords = points[0::2]
    y_coords = points[1::2]
    area = 0.0
    n = len(x_coords)
    for i in range(n):
        j = (i + 1) % n
        area += x_coords[i] * y_coords[j]
        area -= y_coords[i] * x_coords[j]
    area = abs(area) / 2.0
    return area

def calculate_bbox(x_coords, y_coords):
    x_min = min(x_coords)
    y_min = min(y_coords)
    width = max(x_coords) - x_min
    height = max(y_coords) - y_min
    return [x_min, y_min, width, height]


def adjust_polygons(polygons, left, top):
    adjusted_polygons = []
    for polygon in polygons:
        adjusted_polygon = []
        for i in range(0, len(polygon), 2):
            adjusted_polygon.append(polygon[i] + left)
            adjusted_polygon.append(polygon[i+1] + top)
        adjusted_polygons.append(adjusted_polygon)
    return adjusted_polygons



def cvat_to_coco():
    #experimenting of cvat polypoints to coco format. NOT working yet
    img_width = 5304
    img_height = 7952

    # doesn't seem to line up with coco stuff
    #rle cvat
    rle="45, 2, 77, 17, 65, 28, 56, 37, 48, 44, 45, 48, 41, 51, 39, 54, 35, 57, 32, 60, 29, 62, 28, 64, 25, 67, 22, 69, 21, 70, 19, 72, 18, 73, 16, 75, 15, 75, 15, 76, 14, 76, 13, 77, 13, 78, 12, 78, 12, 79, 10, 80, 10, 80, 10, 81, 9, 81, 8, 82, 8, 83, 7, 83, 7, 83, 7, 84, 5, 85, 5, 85, 5, 85, 5, 86, 4, 86, 3, 87, 3, 88, 2, 88, 2, 88, 2, 88, 2, 1888, 1, 89, 1, 89, 1, 89, 1, 89, 1, 89, 1, 89, 1, 89, 1, 89, 1, 89, 1, 89, 1, 89, 1, 89, 1, 89, 1, 89, 1, 89, 1, 88, 2, 88, 3, 87, 3, 87, 3, 86, 4, 86, 4, 86, 5, 85, 5, 84, 6, 84, 6, 84, 7, 83, 7, 82, 8, 82, 8, 82, 9, 80, 10, 80, 10, 80, 10, 79, 12, 78, 12, 78, 12, 78, 12, 77, 14, 76, 14, 76, 14, 75, 16, 74, 16, 73, 17, 73, 18, 72, 18, 71, 20, 69, 21, 69, 22, 66, 25, 64, 26, 62, 29, 59, 32, 57, 33, 55, 36, 52, 39, 49, 42, 46, 48, 40, 54, 33, 62, 26, 68, 19, 77, 11, 85, 2, 43"
    left="1582" 
    top="1024"
    width="90" #matches close enough?
    height="128"
    rle_str = [int(x) for x in rle.split(',')]
    mask = rle_to_binary_mask(rle_str, int(width), int(height), False)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for j, contour in enumerate(contours):
        points = np.squeeze(contour)
        adj_polygon = adjust_polygons(points, int(left), int(top))

    # doesn't seem to line up with coco stuff
    #cvat polgon
    points="1627,1024;1626,1025;1616,1025;1615,1026;1608,1026;1607,1027;1602,1027;1601,1028;1597,1028;1595,1030;1595,1031;1592,1034;1592,1035;1590,1037;1590,1038;1589,1039;1589,1040;1588,1041;1588,1044;1587,1045;1587,1048;1586,1049;1586,1052;1585,1053;1585,1057;1584,1058;1584,1062;1583,1063;1583,1068;1582,1069;1582,1105;1583,1106;1583,1110;1584,1111;1584,1114;1585,1115;1585,1118;1586,1119;1586,1122;1587,1123;1587,1126;1588,1127;1588,1129;1589,1130;1589,1132;1590,1133;1590,1134;1591,1135;1591,1136;1593,1138;1593,1139;1595,1141;1595,1142;1598,1145;1601,1145;1602,1146;1605,1146;1606,1147;1610,1147;1611,1148;1614,1148;1615,1149;1620,1149;1621,1150;1626,1150;1627,1151;1628,1151;1629,1150;1631,1150;1632,1149;1633,1149;1634,1148;1636,1148;1637,1147;1638,1147;1639,1146;1641,1146;1642,1145;1643,1145;1644,1144;1645,1144;1646,1143;1647,1143;1648,1142;1649,1142;1650,1141;1651,1141;1653,1139;1654,1139;1655,1138;1656,1138;1658,1136;1659,1136;1659,1135;1661,1133;1661,1131;1662,1130;1662,1129;1663,1128;1663,1126;1664,1125;1664,1122;1665,1121;1665,1119;1666,1118;1666,1116;1667,1115;1667,1112;1668,1111;1668,1108;1669,1107;1669,1104;1670,1103;1670,1088;1671,1087;1671,1068;1670,1067;1670,1064;1669,1063;1669,1061;1668,1060;1668,1057;1667,1056;1667,1054;1666,1053;1666,1051;1665,1050;1665,1048;1664,1047;1664,1046;1663,1045;1663,1043;1662,1042;1662,1041;1657,1036;1656,1036;1655,1035;1654,1035;1652,1033;1651,1033;1650,1032;1649,1032;1648,1031;1646,1031;1645,1030;1644,1030;1643,1029;1641,1029;1640,1028;1639,1028;1638,1027;1636,1027;1635,1026;1633,1026;1632,1025;1629,1025;1628,1024"
    # points="x0,y0;x1,y1;..."
    points_list = points.split(';')
    points_list = [list(map(float, p.split(','))) for p in points_list]
    points_flat = [coord for point in points_list for coord in point]
    x_coords = points_flat[0::2]
    y_coords = points_flat[1::2]
    bbox = calculate_bbox(x_coords, y_coords) #none of this is correct

    #correct coco result
    #polygon \/
    segmentation_correct = [3945.0,1158.0,3944.0,1159.0,3931.0,1159.0,3930.0,1160.0,3923.0,1160.0,3922.0,1161.0,3921.0,1161.0,3918.0,1164.0,3917.0,1164.0,3915.0,1166.0,3914.0,1166.0,3912.0,1168.0,3911.0,1168.0,3906.0,1173.0,3905.0,1173.0,3903.0,1175.0,3903.0,1176.0,3898.0,1181.0,3898.0,1182.0,3895.0,1185.0,3895.0,1186.0,3894.0,1187.0,3894.0,1188.0,3892.0,1190.0,3892.0,1191.0,3891.0,1192.0,3891.0,1193.0,3890.0,1194.0,3890.0,1204.0,3891.0,1205.0,3891.0,1213.0,3892.0,1214.0,3892.0,1220.0,3893.0,1221.0,3893.0,1225.0,3895.0,1227.0,3896.0,1227.0,3897.0,1228.0,3898.0,1228.0,3899.0,1229.0,3900.0,1229.0,3901.0,1230.0,3902.0,1230.0,3903.0,1231.0,3904.0,1231.0,3905.0,1232.0,3906.0,1232.0,3908.0,1234.0,3909.0,1234.0,3910.0,1235.0,3911.0,1235.0,3914.0,1238.0,3915.0,1238.0,3918.0,1241.0,3919.0,1241.0,3923.0,1245.0,3924.0,1245.0,3925.0,1246.0,3933.0,1246.0,3934.0,1247.0,3941.0,1247.0,3942.0,1248.0,3949.0,1248.0,3950.0,1249.0,3955.0,1249.0,3957.0,1247.0,3958.0,1247.0,3959.0,1246.0,3960.0,1246.0,3962.0,1244.0,3963.0,1244.0,3965.0,1242.0,3966.0,1242.0,3968.0,1240.0,3969.0,1240.0,3971.0,1238.0,3972.0,1238.0,3975.0,1235.0,3976.0,1235.0,3980.0,1231.0,3981.0,1231.0,3986.0,1226.0,3986.0,1224.0,3987.0,1223.0,3987.0,1220.0,3988.0,1219.0,3988.0,1214.0,3989.0,1213.0,3989.0,1207.0,3990.0,1206.0,3990.0,1198.0,3991.0,1197.0,3991.0,1195.0,3990.0,1194.0,3990.0,1192.0,3989.0,1191.0,3989.0,1189.0,3988.0,1188.0,3988.0,1187.0,3987.0,1186.0,3987.0,1185.0,3986.0,1184.0,3986.0,1183.0,3985.0,1182.0,3985.0,1181.0,3976.0,1172.0,3975.0,1172.0,3973.0,1170.0,3972.0,1170.0,3970.0,1168.0,3969.0,1168.0,3967.0,1166.0,3966.0,1166.0,3965.0,1165.0,3964.0,1165.0,3961.0,1162.0,3960.0,1162.0,3957.0,1159.0,3956.0,1159.0,3955.0,1158.0]
    bbox_correct = [3890.0,1158.0,101.0,91.0] #Xtl, Ytl, Habs, Wabs. measured from top left image corner and are 0-index
    area_corect = 7012.0 # = calculate_area(segmentation_correct) !=Habs*Wabs

    import code
    code.interact(local=dict(globals(), **locals()))

    

if __name__ == "__main__":
    main()
    #cvat_to_coco()