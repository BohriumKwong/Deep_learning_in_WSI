# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:44:36 2018

@author: biototem
"""
from __future__ import division
import numpy as np
from PIL import Image
import openslide as opsl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import progressbar
import time
import cv2
import skimage.io as io
import gc
import os,sys
import random
from sklearn.cluster import KMeans
from sklearn.externals import joblib
sys.path.append('/cptjack/totem/StainTools/')
from utils import visual_utils as vu
from utils import misc_utils as mu
from normalization.vahadane import VahadaneNormalizer


def openslide_sample(dir_zero,filename,patch_size,normal_method,half_flag=0):
    save_dir_mean_std = '/cptjack/totem/kmeans_model/mean_std'
    slide = opsl.OpenSlide(dir_zero+ '/IMG/'+filename)
    Wh = np.zeros((len(slide.level_dimensions),2))
    for i in range (len(slide.level_dimensions)):
        Wh[i,:] = slide.level_dimensions[i]
    Ds = np.zeros((len(slide.level_downsamples),2))
    for i in range (len(slide.level_downsamples)):
        Ds[i,0] = slide.level_downsamples[i]
        Ds[i,1] = slide.get_best_level_for_downsample(Ds[i,0]) 
        
    bw_label = cv2.imread(dir_zero+'/label_bw_output/'+os.path.splitext(filename)[0]+'_bw_output.jpg')
    bw_label = cv2.cvtColor(bw_label,cv2.COLOR_BGR2GRAY)
    _, bw_label = cv2.threshold(bw_label,127,1,cv2.THRESH_BINARY)  
    _, contourbs, _, = cv2.findContours(bw_label,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    origin_region_x_num=4
    #将openslide大图在高和宽方向上切成4×4共计16张大的patch进行处理
    origin_region_y_num=4
    w_count = int(Wh[0,0] // patch_size)
    h_count = int(Wh[0,1] // patch_size)
    w_r = w_count//origin_region_x_num
    h_r = h_count//origin_region_y_num
    w_mod = int(Wh[0,0] % patch_size)
    h_mod = int(Wh[0,1] % patch_size)
#    out_img = np.zeros([h_count,w_count])
    px = progressbar.ProgressBar()
    in_num_count = 0
    out_num_count = 0
    half = half_flag*patch_size//2
    for j in px(range(origin_region_x_num * origin_region_y_num)):
        x_r = j % origin_region_x_num #根据j的数值还原当前是哪一行的大pach(0~3)
        y_r = j//origin_region_y_num #根据j的数值还原当前是哪一列的大pach(0~3)
        data_region_in = []
        data_region_out = []
        #有就读图，没有就用read_region的方法读图之后转换再保存，以备后用
        if os.path.exists(dir_zero+'/region/'+os.path.splitext(filename)[0]+'_'+str(j)+'.png'):
            slide_region = io.imread(dir_zero+'/region/'+os.path.splitext(filename)[0]+'_'+str(j)+'.png')
        else:
            slide_region = np.array(slide.read_region((w_mod+patch_size * x_r *w_r,h_mod+patch_size * y_r *h_r),0,(patch_size *w_r,patch_size *h_r)))
            try:
                slide_region1 = normal_method.transform(slide_region[:,:,:3])
                io.imsave(dir_zero+'/region/'+os.path.splitext(filename)[0]+'_'+str(j)+'.png')
            except:
                print('  ' + filename + ' Slide_region Floating point exception.'+' X: '+str(x_r)+' Y: '+str(y_r))
            else:
                slide_region=slide_region1

        for x in range (w_r-half_flag):
            for y in range (h_r-half_flag):
                slide_img = slide_region[y*patch_size+half:(y+1)*patch_size+half,x*patch_size+half:(x+1)*patch_size+half,:]
                rgb_s = (slide_img[:,:,0] >= 200) & (slide_img[:,:,1]>= 200) & (slide_img[:,:,2] >= 200)
                if np.sum(rgb_s)<=(patch_size*patch_size)*0.50:
                    dist_min = 10000
                    x_label= w_mod+(x_r*w_r+x)*patch_size+half
                    y_label= h_mod+(y_r*h_r+y)*patch_size+half
                    for m in range(len(contourbs)):
                        dist = cv2.pointPolygonTest(contourbs[m],(int((x_label+patch_size/2)/Ds[2,0]),int((y_label+patch_size/2)/Ds[2,0])),True)
                        if abs(dist_min) > abs(dist):
                            dist_min = dist
                    if dist_min < -1.5* patch_size / Ds[2,0] :
                        data_mean_std=np.zeros((6))
                        for n in range(3):
                            img = slide_img[:,:,n]
                            data_mean_std[2*n]=img.mean()
                            data_mean_std[2*n+1]=img.std()
                        data_region_out.insert(-1,data_mean_std)
                        out_num_count = out_num_count+1
#                        #if out_num_count%10 == 2 or out_num_count%10 ==5 or out_num_count%10 ==8:
#                        if out_num_count%4==0:
#                            out_dir = test_out_dir
#                        else:
#                            out_dir = train_out_dir
#                        cv2.imwrite(out_dir+'/'+os.path.splitext(filename)[0]+'_x'+str(x_label)+'_y'+str(y_label)+'.jpg',slide_img)
                    elif dist_min > 2.5:
                        data_mean_std=np.zeros((6))
                        for n in range(3):
                            img = slide_img[:,:,n]
                            data_mean_std[2*n]=img.mean()
                            data_mean_std[2*n+1]=img.std()
                        data_region_in.insert(-1,data_mean_std)
                        in_num_count = in_num_count + 1
        if len(data_region_in)>2:
            data_region_in = np.array(data_region_in)
    #        print(data_region)
    #        data_region = data_region.reshape(-1,6)
            k_mean_std = KMeans(n_clusters=2,random_state=1,max_iter=4000,tol=0.00001,algorithm='full').fit(data_region_in)
            joblib.dump(k_mean_std,save_dir_mean_std+'/'+ os.path.splitext(filename)[0] + '_' + str(j) + '_.m')
        if len(data_region_out)>2:
            data_region_in = np.array(data_region_out)
            k_mean_std = KMeans(n_clusters=2,random_state=1,max_iter=4000,tol=0.00001,algorithm='full').fit(data_region_out)
            joblib.dump(k_mean_std,save_dir_mean_std+'/'+ os.path.splitext(filename)[0] + '_' + str(j) + '_out.m')
    return in_num_count,out_num_count
                        
if __name__ == '__main__':
#    bounRange = 2000
    patch_size = 299

    
    dir_zero = '/cptjack/totem/Colon Pathology'
    dir_img = os.listdir(dir_zero+'/IMG/')
    i_0= io.imread('/cptjack/totem/StainTools/18655_.tif')
    normal_method = VahadaneNormalizer()
    normal_method.fit(i_0)
    record = ''
    in_num = 0
    out_num = 0
    p = progressbar.ProgressBar()
    for i in p(range(len(dir_img))):
        in_num_count,out_num_count = openslide_sample(dir_zero,dir_img[i],patch_size,normal_method,0)
        message = str(i) + ' : '+ dir_img[i] + ' in_num_count is: ' + str(in_num_count) + ' , out_num_count is: '+str(out_num_count)
        print('\r\n'+message)
        in_num = in_num + in_num_count
        out_num = out_num + out_num_count
        record = record + message +'\r\n'
    print('All WSL files have been finished!')
    record = record + 'sum of In files = '+ str(in_num) + ', sum of Out files = ' + str(out_num)
    with open('/cptjack/totem/kmeans_data_0_0710.txt','w') as txt_file:
        txt_file.write(record)
        
#    len(data_region) = []
#    data_region.insert(-1,data_mean_std)
#    data_region1 = np.array(data_region)
#    KMeans(n_clusters=2,random_state=1,max_iter=4000,tol=0.00001,algorithm='full').fit(data_region1)
