#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : Fri Mar  1 09:10:45 2018
# @Author  : Kwong
# @Licence : bio-totem
import numpy as np
import math
np.set_printoptions(suppress=True)
def glcm(arr, d_x, d_y, gray_level=16):
    '''计算并返回归一化后的灰度共生矩阵'''
    (height,width)=arr.shape
    max_gray = arr.max() + 1
    #若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    arr = arr * gray_level / max_gray
    ret = np.zeros([gray_level, gray_level])
    for j in range(height - d_y):
        for i in range(width - d_x):
             rows = int(arr[min(j,height-1)][min(i,width-1)])
             cols = int(arr[min(j + d_y,height-1)][min(i + d_x,width-1)])
             ret[rows][cols] = ret[rows][cols]+1
    return ret / float(arr.shape[0] * arr.shape[1]) # 归一化

def glcm_feature(ret,gray_level=16):
    Con=0.0 #对比度
    Ent=0.0 #熵（Entropy, ENT)
    Asm=0.0 #角二阶矩（Angular Second Moment, ASM)
    Idm=0.0 #反差分矩阵（Inverse Differential Moment, IDM)
    for i in range(gray_level):
        for j in range(gray_level):
            Con+=(i-j)*(i-j)*ret[i][j]
            Asm+=ret[i][j]*ret[i][j]
            Idm+=ret[i][j]/(1+(i-j)*(i-j))
            if ret[i][j]>0.0:
                Ent+=ret[i][j]*math.log(ret[i][j])
    return Asm,Con,-Ent,Idm