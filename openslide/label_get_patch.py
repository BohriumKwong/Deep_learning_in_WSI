# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:15:03 2019

@author: biototem
"""

import numpy as np
import openslide as opsl
import os
from skimage import io
import glob

def openslide_cut_patch(filename,label_result_dir,livel,patch_size,save_dir):
    """
    根据给出的label，在以patch size大小的窗口遍历svs原图时对相应区域进行采样
    :param filename:包绝对路径的文件名
    :param label_result_dir:svs图片对应的标注结果存放目录，每个numpy数组大小等同于对应svs图片livel下采样下的大小
    :param livel:svs处理的下采样的级数
    :param patch_size:移动的视野大小，等同于保存图片的大小
    :param save_dir:图片保存的路径
    :return:
    """
    slide = opsl.open_slide(filename)
#    properties = slide.properties
    file_name  = os.path.basename(filename).split('.')[0]
    #获取去除后缀的主文件名
    level_downsamples = slide.level_downsamples[livel]
    Wh = np.zeros((len(slide.level_dimensions),2))
    for i in range (len(slide.level_dimensions)):
        Wh[i,:] = slide.level_dimensions[i]
    w_count = int(Wh[0,0] // patch_size)
    h_count = int(Wh[0,1] // patch_size)

    get_cut = 0
    # 记录采样数量的变量
    
    label_result = np.load(os.path.join(label_result_dir,file_name + '.npy'))
    # 需要注意numpy文件名的逻辑是否是file_name + '.npy'，不是的话自行调整
    # 在执行循环之前，必须确保label_result这个数组是bool数组或者是01二值数组(标注轮廓内的是1，标注轮廓外的是0),如果不是需要自行处理
    for w in range (w_count):
        for h in range (h_count):
            bottom = int(h * patch_size / level_downsamples)
            top = bottom + int(patch_size / level_downsamples) -1
            left = int(w * patch_size / level_downsamples)
            right = left + int(patch_size / level_downsamples) -1                           
            #根据循环的位置推断在2级下采样大小下的label_result对应的起始区域                        
            if np.sum(label_result[bottom : top,left : right ] > 0) > 0.75 * (patch_size / level_downsamples)**2:
                #通过这个判断表明是在轮廓内
                subHIC = np.array(slide.read_region((w * patch_size, h * patch_size), 0, (patch_size, patch_size)))[:,:,:3]
                rgb_s = (abs(subHIC[:,:,0] -107) >= 93) & (abs(subHIC[:,:,1] -107) >= 93) & (abs(subHIC[:,:,2] -107) >= 93)
                if np.sum(rgb_s)<= patch_size**2 * 0.85:
                # 当且仅当白色/黑色像素点加起来不超过read_region截图范围的85%时，才进行保存
                    io.imsave(os.path.join(save_dir,f"{file_name}-{w}_{h}_.png"),subHIC)
                    get_cut += 1
                        
                        


if __name__ == '__main__':
    
    INPUT_IMAGE_DIR = '/cptjack/totem/Lung_animal_model/Test KCI/Right lobe'
    svs_file = glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.ndpi"))
    svs_file = sorted(svs_file)
    LABEL_RESULT_DIR = '/cptjack/totem/zhaofeiyan/image/region/numpy'
    save_dir = '/cptjack/totem/zhaofeiyan/cut_picture'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    patch_size = 512
    livel = 4
    for svs in svs_file:       
        get_cut = openslide_cut_patch(svs,LABEL_RESULT_DIR,livel,patch_size,save_dir)
        print("Finished cutting %d pictures from %s" % (get_cut,os.path.basename(svs)))



                
    
    