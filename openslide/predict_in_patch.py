# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:39:28 2019

@author: biototem
"""

import numpy as np
import openslide as opsl
import os
import cv2
from skimage import io
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from keras import backend as K
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )

#keras_model是在keras训练下的模型，相关的定义和加载不在这里展示
#pytorch_model是在pytorch训练下的模型，相关的定义和加载不在这里展示

def imagenet_processing(image):
    """
    定义一个方法专门用于在使用深度学习模型预测前的图像处理
    :param image:直接在openslide中用read_region读取出来数组化后的图片对象，通道顺序必须确保为RGB
    """
    image = image/255
    ######以下处理为加载imageNet权重才需要的
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        image[:,:,i] -= mean[i]
        image[:,:,i] /= std[i]
    return image


def openslide_predict_patch(filename,livel,keras_model,pytorch_model,imagenet_processing,patch_size):
    """
    根据给出的label，在以patch size大小的窗口遍历svs原图时对相应区域进行采样
    :param filename:包绝对路径的文件名
    :param livel:svs处理的下采样的级数
    :param *_model:已经加载好的深度学习的模型
    :param patch_size:移动的视野大小，等同于保存图片的大小
    :param imagenet_processing: 图像预处理的方法
    :return:
    """
    slide = opsl.open_slide(filename)
#    level_downsamples = slide.level_downsamples[livel]
    w_count = int(slide.level_dimensions[0][0] // patch_size)
    h_count = int(slide.level_dimensions[0][1] // patch_size)
    #设定步长＝patch_size,根据WSI原图大小决定遍历多少个视野框
    out_img = np.zeros([h_count,w_count])
    #根据计算出来的视野框的数目(h*w)定义相同尺寸的数组，每一个元素存放WSI大图中对应位置的patch的预测的结果
    for w in range (w_count):
        for h in range (h_count):
            subHIC = np.array(slide.read_region((w * patch_size, h * patch_size), 0, (patch_size, patch_size)))[:,:,:3]
            rgb_s = (abs(subHIC[:,:,0] -107) >= 93) & (abs(subHIC[:,:,1] -107) >= 93) & (abs(subHIC[:,:,2] -107) >= 93)
            if np.sum(rgb_s)<= patch_size**2 * 0.64:
            #一般来说需要加个颜色判断，上面语句就是统计一下偏白或者偏黑的像素点，不超过一定占比的时候才进行下一步处理
                subHIC = imagenet_processing(subHIC)
                #调用已经定义好的imagenet_processing方法对图像进行预处理
                slide_img = np.expand_dims(subHIC, axis=0)
                #必须要增加一个维度，形成4D张量才能送进模型预测，也可以将多张图叠加为4D向量才进行预测
                
                ################以下是使用keras_model进行调用的方法###################
                out = keras_model.predict(slide_img)
                
                ################以下是使用pytorch_model进行调用的方法###################
                slide_img = Variable(torch.from_numpy(slide_img.transpose((0,3, 1, 2))).float().cuda())
                torch.no_grad()
                prob = pytorch_model(slide_img)
                region_predict = F.softmax(prob )
                out = region_predict.cuda().data.cpu().numpy()
                
                out_img[h,w] = int(np.argmax(out,axis=0)) + 1
                # 因为创建的是全零矩阵，为了将预测的标签中的0和背景的0区分起来，对预测的结果进行+1处理
                
                
    return out_img