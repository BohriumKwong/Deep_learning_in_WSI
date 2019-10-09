# @Time    : 2019.09.05
# @Author  : Bohrium.Kwong
# @Licence : bio-totem


import os
import shutil
import lxml.etree as ET
import cv2
from PIL import ImageDraw, Image
import numpy as np


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
        if color.attrib['LineColor']=='65280':
            for region in tree.findall('.//Annotation/Regions/Region'):
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
            x_max = max(point_list, key=lambda point: point[0])[0]
            x_min = min(point_list, key=lambda point: point[0])[0]
            y_max = max(point_list, key=lambda point: point[1])[1]
            y_min = min(point_list, key=lambda point: point[1])[1]
            # mislabeled, here checked by x and y coordinate max and min difference
            if (x_max - x_min < 50) or (y_max - y_min < 50): continue
            if region_class[r_class] == '3':
                dr.arc(point_list, 0, 360, fill='#000000', width=12)
            else:
                dr.line(point_list, fill="#000000", width=12)                            

    return im

def region_binary_image(tile, region_list,region_class, level_downsample):
    """
    convert the region labeled or not by doctor to binary image
    :param tile: a return image based on the method of Slide class object in 'utils.openslide_utils'
    :param region_list: region list, region point,
                    eg : [[{'X': '27381.168113', 'Y': '37358.653791'}], [{'X': '27381.168113', 'Y': '37358.653791'}]]
    :param region_class : list,keep the value of region.attrib.get('Type') in elements of region list
                    eg : [0,0,0,1,2,3]
    :param level_downsample: slide level down sample
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
        if dist(p_list[0], p_list[-1]) < 150 and j not in pin_jie_flag:
        #如果收尾坐标距离相差在150范围内(曼哈顿距离)，且未成被拼接过，直接认为这个组坐标无须拼接，存储起来
            single_list.append(p_list)                
        elif dist(p_list[0], p_list[-1]) > 150 and j not in pin_jie_flag:
        #如果收尾坐标距离相差在150范围外(曼哈顿距离)，且未成被拼接过，说明这组坐标是残缺非闭合的，需要对其余标注坐标进行新一轮的循环判断
            for j_2,p_list_2 in enumerate(regions_list):
                while j_2 != j and dist(p_list[-1],p_list_2[0]) < 150 and j_2 not in pin_jie_flag:
                    p_list = p_list + p_list_2.copy()
                    # 当这组非闭合的尾坐标和其他组坐标的首坐标接近到一定范围时(距离是150内),就让当前的非闭合的坐标列表和该组坐标列表相加
                    pin_jie_flag.append(j_2)
                    # 处理完毕之后，将该组坐标的序号增加到已拼接坐标的列表中，确保后续循环不会再判断这个列表
            single_list.append(p_list)
            
    for points in single_list:
        dr.polygon(points, fill="#ffffff")
            
    #由于医生的标注除了出现不连续(非闭合)的情况外，还存在多余勾画的情况，对这种情况暂时没有完整的思路予以接近，先用
    # opencv中的开闭操作组合来进行修补
    kernel = np.ones((20,20),np.uint8)                                
    filter_matrix = np.array(im).astype(np.uint8)
    filter_matrix = cv2.morphologyEx(filter_matrix, cv2.MORPH_OPEN, kernel)
    filter_matrix = cv2.morphologyEx(filter_matrix, cv2.MORPH_CLOSE, kernel)  
#    plt.imshow(filter_matrix)              
    return filter_matrix



from shapely import affinity
from shapely.geometry import Point, Polygon, LinearRing


class PointPolygon:
    def point(self, coordinate):
        """
        generate shapely point object
        :param coordinate: coordinate, python tuple: (x, y)
        :return:
        """

        return Point(coordinate)

    def polygon(self, coordinates):
        """
        generate shapely polygon object
        :param coordinates:  list of coordinate, [(x1, y1), (x2, y2)]
        :return:
        """

        return Polygon(coordinates)

    def ellipse(self, point1, point2):
        """
        generate ellipse polygon object

        ellipse = ((x_center, y_center), (a, b), angle):
            (x_center, y_center): center point (x,y) coordinates,
            (a,b): the two semi-axis values (along x, along y),
            angle: angle in degrees between x-axis of the Cartesian base and the corresponding semi-axis.

        :param point1: the point to inscribed ellipse
        :param point2: the point to inscribed ellipse
        :return: ellipse of polygon object
        ref: https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely
        """

        x_center, y_center = (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2
        a, b = abs(point1[0] - point2[0]) / 2, abs(point1[1] - point2[1]) / 2
        ellipse = ((x_center, y_center), (a, b), 90)
        circ = Point(ellipse[0]).buffer(1)
        ell = affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))
        elrv = affinity.rotate(ell, 90 - ellipse[2])

        return elrv

    def polygon_bound(self, polygon):
        """
        get the boundary of the polygon, that is LinearRing
        :param polygon: polygon
        :return:
        """

        return LinearRing(list(polygon.exterior.coords))

    def point_pylygon_position(self, point, polygon):
        """
        check the point is inside or outside the polygon
        :param point: the point
        :param polygon: the polygon
        :return: true -- point inside polygon, false -- point outside polygon (include the boundary)
        """

        return point.within(polygon)

    def point_polygon_distance(self, point, polygon):
        """
        get the distance from point to polygon
        :param point: the point
        :param polygon: the polygon
        :return:
        """

        linearRing = self.polygon_bound(polygon)

        return point.distance(linearRing)


def point_position(coordinate, region):
    """
    check the point is inside or outside the region
    :param coordinate: the point to be checked, python tuple: (X, Y)
    :param region: the region, python list: [(X1, Y1), (X2, Y2)]
    :return: (boolean, float) -- > (false[outside] / true[inside], distance))
    """

    point_polygon = PointPolygon()
    point = point_polygon.point(coordinate)
    polygon = point_polygon.polygon(region)
    position = point_polygon.point_pylygon_position(point, polygon)
    distance = point_polygon.point_polygon_distance(point, polygon)

    return position, distance

