# @Time    : 2019.10.18
# @Author  : Bohrium.Kwong
# @Licence : bio-totem


import os
import shutil
import lxml.etree as ET

from PIL import ImageDraw, Image
import numpy as np
import copy
import cv2

current_path = os.path.dirname(__file__)

def dist(a, b):
    return round(abs(a[0]-b[0]) + abs(a[1]-b[1]))

def xml_to_region(xml_file):
    """
    parse XML label file and get the points
    :param xml_file: xml file
    :return: region list,region_class
    """

    tree = ET.parse(xml_file)
    region_list = []
    region_class = []
    for color in tree.findall('.//Annotation'):
        if color.attrib['LineColor'] in ['65280','255','65535']:
            # '65280'是绿色,'255'是红色,可以根据自己的实际情况更改这个判断条件(或者直接if True)
            for region in color.findall('./Regions/Region'):
                vertex_list = []
                #region.attrib.get('Type')=='0':
                region_class.append(region.attrib.get('Type'))
                for vertex in region.findall('.//Vertices/Vertex'):
                    # parse the 'X' and 'Y' for the vertex
                    vertex_list.append(vertex.attrib)
                region_list.append(vertex_list)

    return region_list,region_class


def region_handler(im, region_list,region_class, level_downsample):
    """
    handle region label point to discrete point, and draw the region point to line
    :param im: the image painted in region line
    :param region_list: region list, region point,
                    eg : [[{'X': '27381.168113', 'Y': '37358.653791'}], [{'X': '27381.168113', 'Y': '37358.653791'}]]
    :param region_class : list,keep the value of region.attrib.get('Type') in elements of region list
                    eg : [0,0,0,1,2,3]
    :param level_downsample: slide level down sample
    :return: image painted in region line of numpy array format
    """
    
    dr = ImageDraw.Draw(im)
    for r_class, region in enumerate(region_list):
        point_list = []
        if region_class[r_class] == '0' or region_class[r_class] == '3':
            for __, point in enumerate(region):
                X, Y = int(float(point['X'])/level_downsample), int(float(point['Y'])/level_downsample)
                point_list.append((X, Y))

#        points_length = len(point_list)
#            x_max = max(point_list, key=lambda point: point[0])[0]
#            x_min = min(point_list, key=lambda point: point[0])[0]
#            y_max = max(point_list, key=lambda point: point[1])[1]
#            y_min = min(point_list, key=lambda point: point[1])[1]
            # mislabeled, here checked by x and y coordinate max and min difference
            #if (x_max - x_min < 50) or (y_max - y_min < 50): continue
        ## 上述这个逻辑很容易过滤小轮廓而不显示，暂且将其注释掉. ——by Bohrium Kwong 20201125
            
            if region_class[r_class] == '3':
                dr.arc(point_list, 0, 360, fill='#000000', width=12)
            else:
                dr.line(point_list, fill="#000000", width=12)                            

    return im

def region_binary_image(tile, region_list,region_class, level_downsample,label_correction = True):
    """
    convert the region labeled or not by doctor to binary image
    :param tile: a return image based on the method of Slide class object in 'utils.openslide_utils'
    :param region_list: region list, region point,
                    eg : [[{'X': '27381.168113', 'Y': '37358.653791'}], [{'X': '27381.168113', 'Y': '37358.653791'}]]
    :param region_class : list,keep the value of region.attrib.get('Type') in elements of region list
                    eg : [0,0,0,1,2,3]
    :param level_downsample: slide level down sample
    :param label_correction: label correctting or not 
    :return: image painted in region line of numpy array format
    """
    im = Image.new(mode="1", size=tile.size)
    dr = ImageDraw.Draw(im)
    regions_list = []
    for r_class, region in enumerate(region_list):
        point_list = []
        if region_class[r_class] == '0':
            for __, point in enumerate(region):
                X, Y = int(float(point['X'])/level_downsample), int(float(point['Y'])/level_downsample)
                point_list.append((X, Y))

            regions_list.append(point_list)
    
    if label_correction:
    # 考虑到有些读取xml的场景是针对分割生成的结果，有一些非常小的区域，故在这里新增一个label_correction参数，只有其值为True的时候才执行修正
    #   ——by Bohrium.kwong 2020.11.25
    
        #由于医生的标注会出现不连续(非闭合)的情况，导致提取出来的标注坐标列表，会分成多段，比如：
        #            0 (1979, 798) (2144, 1479)
        #            1 (2139, 1483) (2319, 2162)
        #            2 (2308, 2160) (3003, 1646)
        # 正常情况下，标注坐标列表应该收尾闭合(前后坐标一致)，上下列表之间差异应该较大，如：
        #            12 (1177, 2986) (1177, 2986)
        #            13 (1507, 2942) (1507, 2940)
        # 针对上述第一种情况，需要对提出出来的标注坐标列表进行循环判断，对收尾不一而且和其他标注列表的首坐标相对较近的话，进行合并处理
        
        pin_jie_flag = [] #存储已经被拼接过的标注坐标列表序号                  
        single_list = [] #存储新标注坐标列表的列表          
        for j,p_list in enumerate(regions_list):
            if dist(p_list[0], p_list[-1]) < 50 and j not in pin_jie_flag:
            #如果首尾坐标距离相差在150范围内(曼哈顿距离)，且未成被拼接过，直接认为这个组坐标无须拼接，存储起来
                single_list.append(p_list)                
            elif dist(p_list[0], p_list[-1]) > 50 and j not in pin_jie_flag:
            #如果首尾坐标距离相差在150范围外(曼哈顿距离)，且未成被拼接过，说明这组坐标是残缺非闭合的，需要对其余标注坐标进行新一轮的循环判断
                for j_2,p_list_2 in enumerate(regions_list):
                    if j_2 != j and j_2 not in pin_jie_flag:
    
                        if dist(p_list[-1],p_list_2[0]) < 50 :
                            p_list = p_list + p_list_2.copy()
                            pin_jie_flag.append(j_2)
                        elif dist(p_list[0],p_list_2[-1]) < 50 :
                            p_list = p_list_2.copy() + p_list
                            pin_jie_flag.append(j_2)
                        elif dist(p_list[-1],p_list_2[-1]) < 50 :
                            p_list_2_new = copy.deepcopy(p_list_2)
                            p_list_2_new.reverse()
                            p_list = p_list + p_list_2_new
                            pin_jie_flag.append(j_2)
                        elif dist(p_list[0],p_list_2[0]) < 50 :
                            p_list_2_new = copy.deepcopy(p_list_2)
                            p_list_2_new.reverse()
                            p_list = p_list_2_new + p_list
                            pin_jie_flag.append(j_2)
                        # 当这组非闭合的尾坐标和其他组坐标的首坐标接近到一定范围时(距离是150内),就让当前的非闭合的坐标列表和该组坐标列表相加                        
                        # 处理完毕之后，将该组坐标的序号增加到已拼接坐标的列表中，确保后续循环不会再判断这个列表
                single_list.append(p_list)
        for points in single_list:
            dr.polygon(points, fill="#ffffff")
            
        #由于医生的标注除了出现不连续(非闭合)的情况外，还存在多余勾画的情况，对这种情况暂时没有完整的思路予以接近，先用
        # opencv中的开闭操作组合来进行修补
        kernel = np.ones((15,15),np.uint8)                                
        filter_matrix = np.array(im).astype(np.uint8)
        filter_matrix = cv2.morphologyEx(filter_matrix, cv2.MORPH_OPEN, kernel)
        filter_matrix = cv2.morphologyEx(filter_matrix, cv2.MORPH_CLOSE, kernel)  

    else:
        for points in regions_list:
            dr.polygon(points, fill="#ffffff")
        filter_matrix = np.array(im).astype(np.uint8)
#    plt.imshow(filter_matrix)              
    return filter_matrix


class Region:
    """"
    handle the template xml format file to insert label svs region
    """
    def __init__(self, xml_file):
        parser = ET.XMLParser(remove_blank_text=True)
        if not os.path.isfile(xml_file):
            template = os.path.join(current_path, "template.xml")
            shutil.copy(template, xml_file)
        self._xml_file = xml_file
        self._tree = ET.parse(xml_file, parser)

    def get_region(self, region_id):
        """
        get region by region id
        :param region_id: region id, 0: green, 1: yellow, 2: red, see the template.xml
        :return: the region
        """

        return self._tree.findall(".//Annotation/Regions")[region_id]

    def add(self, region_id, points):
        """
        add one region to the specified region by region id, the added region is ellipse
        and the parameter points is a rectangle bounded by an ellipse
        :param points: list with two element(upper-left, bottom-right), is the rectangle bounded by an ellipse
        :return:
        """

        region = self.get_region(region_id)
        region_num = len(region.findall(".//Region"))
        region_attr = {
            "Id": str(region_num+1),
            "Type": "2",
            "Zoom": "1",
            "Selected": "1",
            "ImageLocation": "",
            "ImageFocus": "0",
            "Length": "80",
            "Area": "400",
            "LengthMicrons": "20",
            "AreaMicrons": "30",
            "Text": "",
            "NegativeROA": "0",
            "InputRegionId": "0",
            "Analyze": "0",
            "DisplayId": "1"
        }

        region_tag = ET.Element("Region", region_attr)
        region.append(region_tag)

        attributes = ET.SubElement(region_tag, "Attributes")
        vertices = ET.Element("Vertices")
        region_tag.append(vertices)

        for point in points:
            # insert point
            ET.SubElement(vertices, "Vertex", attrib=point)

    def save(self):
        """
        save the xml file
        :return:
        """

        self._tree.write(self._xml_file, pretty_print=True)
        

def color_int_to_str(color_int):
    color_str = hex(color_int)[2:]
    assert len(color_str) <= 6, 'Found unknow color!'
    pad_count = 6 - len(color_str)
    color_str = ''.join(['0'] * pad_count) + color_str
    b, g, r = color_str[0:2], color_str[2:4], color_str[4:6]
    return r+g+b


def color_str_to_int(color_str):
    assert len(color_str) == 6, 'Found unknow color!'
    r, g, b = color_str[0:2], color_str[2:4], color_str[4:6]
    color_int = (int(r, 16)) + (int(g, 16) << 8) + (int(b, 16) << 16)
    return color_int


def contours_to_xml(savepath,contours,if_add = False,level_downsample = 16,mpp= "0.252100",linecolor ="16711680",contour_area_threshold=2000):
    """
    based on a mask of svs file(mask sure the size of the mask equals the size of the svs file 's level_dimensions in level 2) to make a
    xml format lable file for this svs file
    :param savepath :  the xml format lable file save file path
    :param contours :  contours list return from cv2.findContours of the mask
    :param if_add : Added niew Annotation to an exits xml format label file or not ,defaut False
    :param level_downsample : the value of slide.level_downsamples[2]
    :param mpp : the value of MicronsPerPixel in slide.properties['openslide.mpp-x']
    :param linecolor : the value of decimal color code to draw contours in xml format lable file ,default color is blue
    :param contour_area_threshold : the threshold to drop small contours base on cv2.contourArea,which helps to keep the big area contours in xml format lable  file
    :return:
    """
    ann_begin_tag = 1
    Annotations = ET.Element('Annotations', {'MicronsPerPixel': mpp})
    origin_color_list = []
    if if_add and os.path.exists(savepath):
        origin = ET.parse(savepath)
        ann_begin_tag = len(origin.findall('.//Annotation')) + 1
        for ann in origin.findall('.//Annotation'):
            origin_color_list.append(ann.attrib['LineColor'])
            Annotations.append(ann)
                               
    if linecolor in origin_color_list: linecolor = "13382297"
    Annotation = ET.SubElement(Annotations, 'Annotation',
                                          {'Id': str(ann_begin_tag), 'Name': '', 'ReadOnly': '0', 'NameReadOnly': '0',
                                           'LineColorReadOnly': '0', 'Incremental': '0', 'Type': '4',
                                           'LineColor': linecolor, 'Visible': '1', 'Selected': '1',
                                           'MarkupImagePath': '', 'MacroName': ''})
    Attributes = ET.SubElement(Annotation, 'Attributes')
    ET.SubElement(Attributes, 'Attribute', {'Name': '', 'Id': '0', 'Value': ''})
    Regions = ET.SubElement(Annotation, 'Regions')
    RegionAttributeHeaders = ET.SubElement(Regions, 'RegionAttributeHeaders')
    ET.SubElement(RegionAttributeHeaders, 'AttributeHeader',
                             {'Id': "9999", 'Name': 'Region', 'ColumnWidth': '-1'})
    ET.SubElement(RegionAttributeHeaders, 'AttributeHeader',
                             {'Id': "9997", 'Name': 'Length', 'ColumnWidth': '-1'})
    ET.SubElement(RegionAttributeHeaders, 'AttributeHeader',
                             {'Id': "9996", 'Name': 'Area', 'ColumnWidth': '-1'})
    ET.SubElement(RegionAttributeHeaders, 'AttributeHeader',
                             {'Id': "9998", 'Name': 'Text', 'ColumnWidth': '-1'})
    ET.SubElement(RegionAttributeHeaders, 'AttributeHeader',
                             {'Id': "1", 'Name': 'Description', 'ColumnWidth': '-1'})
    i = 1
    for cnt in contours:
        contour_area = cv2.contourArea(cnt)
        if contour_area > contour_area_threshold:
            Region = ET.SubElement(Regions, 'Region',
                                          {'Id': str(i), 'Type': '0', 'Zoom': '0.011', 'Selected': '0',
                                           'ImageLocation': '', 'ImageFocus': '-1', 'Length': str(cnt.shape[0]), 'Area': str(level_downsample**2*contour_area),
                                           'LengthMicrons': '0', 'AreaMicrons': '0', 'Text': '', 'NegativeROA': '0',
                                           'InputRegionId': '0', 'Analyze': '1', 'DisplayId': str(i)})
            ET.SubElement(Region, 'Attributes')
            Vertices = ET.SubElement(Region, 'Vertices')
            cnt = np.squeeze(np.asarray(cnt))
            for j in range(cnt.shape[0]):
                ET.SubElement(Vertices, 'Vertex', {'X': str(cnt[j,0]*level_downsample), 'Y': str(cnt[j,1]*level_downsample)})
            i = i + 1
    ET.SubElement(Annotation, 'Plots')
    doc = ET.ElementTree(Annotations)
    doc.write(open(savepath, "wb"), pretty_print=True)

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from utils.openslide_utils import Slide
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = 15, 15
    slide = Slide('your svs file path/filename.svs')
#    xml_file = 'your svs label file path'
#    tile = slide.get_thumb()#获取2级采样下的全片截图
#    region_list,region_class= xml_to_region(xml_file)
    # 在这里使用xml_utils的方法进行指定区域提取(最终返回的是个True False矩阵)
#    region_process_mask = region_binary_image(tile, region_list, region_class,slide.get_level_downsample())
#    # 根据上述返回的标注坐标列表生成WSI原图2级采样大小的True False矩阵
#    region_label = region_handler(tile, region_list, slide.get_level_downsample())
#    plt.imshow()
#    plt.imshow(region_process_mask)
    region_process_mask = np.load('your svs mask result file path')
    if cv2.__version__[0]=='4':
        cnts, _ = cv2.findContours(region_process_mask,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, cnts , _ = cv2.findContours(region_process_mask,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
        
    savepath = 'your svs file path/filename.xml'
    mpp = str(slide.get_mpp()*1000)
    level_downsample = slide.get_level_downsample()
    contours_to_xml(savepath,cnts,True)
