# @Time    : 2019.03.01
# @Author  : kawa Yeung
# @Licence : bio-totem


import cv2
import numpy as np
import sys
sys.path.append('../')
from utils.opencv_utils import OpenCV


def get_tissue(im, contour_area_threshold):
    """
    Get the tissue contours from image(im)
    :param im: numpy 3d-array object, image with RGB mode
    :param contour_area_threshold: python integer, contour area threshold, tissue contour is less than it will omit
    :return: tissue_cnts: python list, tissue contours that each element is numpy array with shape (n, 2)
    """

    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    opencv = OpenCV(im)
#    binary = opencv.gray_binary(thresh=230, show=False)
#    morphology = opencv.erode_dilate(binary, erode_iter=0, dilate_iter=3, show=False)
    cnts = opencv.find_contours(is_erode_dilate = True)
    tissue_cnts = []

    for each, cnt in enumerate(cnts):
        contour_area = cv2.contourArea(cnt)
        if contour_area < contour_area_threshold:
            # omit the small area contour
            del cnts[each]
            continue
        tissue_cnts.append(np.squeeze(np.asarray(cnt)))

    # initialize mask to zero
    mask = np.zeros((im.shape[0], im.shape[1])).astype(im.dtype)
    color = [1]
    mask = cv2.fillPoly(mask, cnts, color)

    return mask, tissue_cnts


if __name__ == "__main__":
    from skimage import io
    import matplotlib.pyplot as plt
    im = io.imread("/Users/kawa/Desktop/52800/52800_.png")
    im = im.astype(np.uint8)
    mask, _ = get_tissue(im, contour_area_threshold=10000)
    plt.imshow(mask)
    plt.show()
