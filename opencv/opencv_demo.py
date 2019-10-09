#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# ### 使用opencv读入图片，通道顺序默认是BGR，和我们所习惯的RGB不一样。在处理是要注意通道顺序，或者先对其进行颜色空间转换。但同时，如果使用opencv的imwrte保存图像时，默认图像是BGR的，因此如果要使用该方法保存图像时，如果图像通道顺序RGB的话，还得再转换过来

# In[2]:


svs_img = cv2.imread('/cptjack/totem/Colon Pathology/preview_black/18686__preview.png')
plt.rcParams['figure.figsize'] = 15, 15
plt.imshow(svs_img)
# matplotlib默认图像是RGB空间的，因此直接imshow用opencv读取的图像时，显示的颜色是不正确的。


# In[3]:


svs_img = cv2.cvtColor(svs_img, cv2.COLOR_BGR2RGB)
plt.imshow(svs_img)


# In[4]:


# 以下方法是我们此前用来对svs缩略图进行背景组织区域分离的方法
def get_tissue(im, contour_area_threshold):
    """
    Get the tissue contours from image(im)
    :param im: numpy 3d-array object, image with RGB mode
    :param contour_area_threshold: python integer, contour area threshold, tissue contour is less than it will omit
    :return: tissue_cnts: python list, tissue contours that each element is numpy array with shape (n, 2)
    """

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    binary = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#     morphology = cv2.erode(binary, kernel, iterations = 0)
#  erode是opencv中膨胀的操作 
#     morphology = cv2.dilate(morphology, kernel, iterations = 3)
#  dilate是opencv中腐蚀的操作
    morphology = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# cv2.morphologyEx指定cv2.MORPH_OPEN参数就是进行开操作
    _, cnts, _ = cv2.findContours(morphology.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 对处理后的二值图像进行轮廓提取，并过滤掉轮廓面积小于设定阈值的轮廓
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
    return mask, cnts


# In[5]:


mask, cnts = get_tissue(svs_img, contour_area_threshold=1000)
plt.imshow(mask)


# In[6]:


svs_img_new = svs_img.copy()
for i,contour in enumerate(cnts):
    if cv2.contourArea(cnts[i]) > 1000:
        cv2.drawContours(svs_img_new,cnts,i,(76,177,34),15)
        # 轮廓着色的方法
plt.imshow(svs_img_new)

