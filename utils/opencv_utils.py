# @Time    : 2019.10.15
# @Author  : Bohrium.Kwong
# @Licence : bio-totem


import cv2
import numpy as np

# Based on opencv-python==4.1.0.25
class OpenCV:
    def __init__(self, im):
        self._im = im

    def gray_binary(self, thresh=200, show=False,binary_arg = cv2.THRESH_BINARY_INV):
        """
        convert image to binary image after gray scale
        :param thresh: binary image threshold
        :param show: show the binary dynamically
        :param binary_arg:cv2.THRESH_BINARY(0) or cv2.THRESH_BINARY_INV(1)
        :return:
        """
        if len(self._im.shape) >2 :
            gray = cv2.cvtColor(self._im, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.uint8(self._im)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        binary = cv2.threshold(blurred, thresh, 255, binary_arg)[1]

        def binary_theshold(threshold):
            self.binary = cv2.threshold(blurred, threshold, 255, binary_arg)[1]
            cv2.imshow("binary image", self.binary)

        if show:
            self.window_name = "binary image"
            cv2.namedWindow(self.window_name, 0)
            cv2.resizeWindow(self.window_name, 640, 480)
            cv2.createTrackbar("binary threshold", self.window_name, 100, 255, binary_theshold)
            binary_theshold(100)
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()

        return binary

    def erode_dilate(self, binary, erode_iter=6, dilate_iter=9, kernel_size=(5, 5), show=False):
        """
        errode and dilate the binary image dynamically
        :param binary: binary image, output of self.gray_binary
        :param erode_iter: erode iteration, default 2
        :param dilate_iter: dilate iteration, default 3
        :return:
        """

        morphology = binary

        if show:
            self.window_name = "errode dilate"
            cv2.namedWindow(self.window_name, 0)
            cv2.resizeWindow(self.window_name, 640, 480)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        while show:
            cv2.imshow(self.window_name, morphology)
            key = cv2.waitKey(1) & 0xFF
            if ord('e') == key:
                morphology = cv2.erode(morphology, kernel, iterations=1)
                print('erode')
            if ord('d') == key:
                morphology = cv2.dilate(morphology, kernel, iterations=1)
                print('dilate')
            if ord('r') == key:
                morphology = binary
                print('reset threshold image')
            if ord('q') == key:
                break

        cv2.destroyAllWindows()

        morphology = cv2.erode(morphology, kernel, iterations=erode_iter)
        morphology = cv2.dilate(morphology, kernel, iterations=dilate_iter)

        return morphology
    
#    if is_morph is not None and is_morph in ['dilate','erode','open']:
    
    def morphologyex_process(self,is_binary = True,kernel = np.ones((3, 3), np.uint8),method = 'dilate',iterations=3):
        if not is_binary:
            morphology = self.gray_binary(thresh=200, show=False)
        else:
            morphology = self._im
        
        if method is not None and method in ['dilate','erode','open','close']:
            if method == 'dilate':
                morphology = cv2.dilate(morphology, kernel, iterations = iterations)
            elif method == 'erode':
                morphology = cv2.erode(morphology, kernel, iterations = iterations)
            elif method == 'open':
                morphology = cv2.morphologyEx(morphology, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif method == 'close':
                morphology = cv2.morphologyEx(morphology, cv2.MORPH_CLOSE, kernel, iterations=iterations)        
        return morphology               

            
    def find_contours(self,is_erode_dilate = False,is_hier_return = False,is_morphologyex = None,iterations=3,is_binary=True,mode= 3):
        """
        get the points of contours
        :param morphology: output of self.erode_dilate
        :return:
        """
        if not is_binary:
            morphology = self.gray_binary(thresh=200, show=False)
        else:
            morphology = self._im
            
        if is_erode_dilate:
            morphology = self.erode_dilate(morphology, erode_iter=0, dilate_iter=3, show=False)
        morphology = cv2.copyMakeBorder(morphology,0,0,0,0,cv2.BORDER_CONSTANT,value=0)


        if is_morphologyex is not None and is_morphologyex in ['dilate','erode','open','close']:
            morphology = self.morphologyex_process(method = is_morphologyex,iterations = iterations)
        
        # 这里传参要设置成int类型
        # https://docs.opencv.org/4.1.0/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
        # https://docs.opencv.org/4.1.0/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
        if not isinstance(mode,int):
            mode_dic = {'RETR_EXTERNAL':cv2.RETR_EXTERNAL, 'RETR_LIST':cv2.RETR_LIST, 'RETR_CCOMP':cv2.RETR_CCOMP, 'RETR_TREE':cv2.RETR_TREE, 'RETR_FLOODFILL':cv2.RETR_FLOODFILL}
            mode = mode_dic[mode]
        # 本来指定使用版本是4.x，但考虑到很多人用的依旧是3.X,所以针对findContours的用法对两个版本都做了兼容处理.By Kwong 20190730
        if cv2.__version__[0]=='4':
            cnts, hier = cv2.findContours(morphology,mode=mode,method=cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, cnts , hier = cv2.findContours(morphology,mode=mode,method=cv2.CHAIN_APPROX_SIMPLE)

        if not is_hier_return:
            return cnts
        else:
            return cnts , hier

    def draw_contours(self, cnts):
        """
        draw the contours
        :param cnts: output of self.find_contours
        :return:
        """

        im = cv2.drawContours(self._im, cnts, -1, (0, 255, 0), 1)

        self.window_name = "find contours"
        cv2.namedWindow(self.window_name, 0)
        cv2.resizeWindow(self.window_name, 640, 480)
        cv2.imshow(self.window_name, im)
        cv2.waitKey(0)

    def extract_contours(self, cnts, show=False):
        """
        extract contours from origin image
        :param cnts: output of self.find_contours
        :return:
        """

        mask = np.zeros(self._im.shape).astype(self._im.dtype)    # all black
        color = [255, 255, 255]
        cv2.fillPoly(mask, cnts, color)
        result = cv2.bitwise_and(self._im, mask)

        if show:
            self.window_name = "extract contours"
            cv2.namedWindow(self.window_name, 0)
            cv2.resizeWindow(self.window_name, 640, 480)
            cv2.imshow(self.window_name, result)
            cv2.waitKey(0)
            
            
    def resize(self,resize_shape_w,resize_shape_h,interpolation = 'default'):
        if interpolation == 'default':
            if self._im.shape[1] < 100 and self._im.shape[0] < 100:
                im_resize = cv2.resize(self._im,(int(resize_shape_w),int(resize_shape_h)), interpolation=cv2.INTER_NEAREST)
            elif resize_shape_w < self._im.shape[1] and resize_shape_h < self._im.shape[0]:
                im_resize = cv2.resize(self._im,(int(resize_shape_w),int(resize_shape_h)), interpolation=cv2.INTER_AREA)
            else:
                im_resize = cv2.resize(self._im,(int(resize_shape_w),int(resize_shape_h)), interpolation=cv2.INTER_CUBIC)
        else:
            if not isinstance(interpolation,int):
                interpolation_dic = {'INTER_NEAREST': 0, 'INTER_LINEAR': 1, 'INTER_CUBIC': 2, 'INTER_AREA': 3,
                                     'INTER_LANCZOS4': 4,
                                     'INTER_LINEAR_EXACT': 5, 'INTER_MAX': 7, 'WARP_FILL_OUTLIERS': 8,
                                     'WARP_INVERSE_MAP': 16}
                interpolation = interpolation_dic[interpolation]
            im_resize = cv2.resize(self._im, (int(resize_shape_w), int(resize_shape_h)), interpolation=interpolation)
        return im_resize