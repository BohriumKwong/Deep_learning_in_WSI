# Openslide及OpenCV使用入門資料


个人从事WSI医学病理图像分析，有感于将深度学习用于病理图像分析时，涉及数据处理方面的步骤非常多，现将常用的库包及方法进行整理。

##  Installation

```pip install -r requirements.txt```


## 目录说明

### ./opencv/

该目录是提供几个我在项目中使用opencv python接口比较常用的操作，如形态学操作，颜色空间转换等。其中还有一些提取图片底层特征分析的demo(包括自己实现的灰度共生矩阵特征向量的提取)。


上述代码包括py脚本文件和jupyter notebook的ipynb文件,另外针对python opencv接口的使用还会提供电子书。需要注意的是该电子书是基于3.X版本，4.X版本在使用轮廓提取的方法时，返回的变量稍有差异。

### ./openslide/
请仔细阅读**./openslide/README.md**文档。

#### openslide_demo.py
主要提供了openslide基本的用法，如常用到的get_thumbnail和read_region，该代码包括py脚本文件和jupyter notebook的ipynb文件，方便查阅。

#### label_get_patch.py
演示了基于指定的label(mask形式)，对WSI图片使用openslide的read_region的方法进行全局循环来采样的demo。实际上，只要掌握这个方法，以后相关的处理任务都能用同样的思路进行(包括做深度学习方面的预测)。

#### big_patch_transfrom_samping.py
另一种对WSI图片进行采样的方法，现在已经比较少用。因为在这个场景需要使用Stain_tools中的normalization包下的V方法进行颜色标准化，如果多次需要用到颜色标准化后的图像的话，可直接先对大图区域进行一次性的颜色标准化(比如在WSI图片0级下采样下分割成4×4的区域),标准化完成之后，保存该转换后的大切割图，以后可以反复使用该图进行后续处理(包括采样和预测)。这样做可以比对逐个patch进行标准化更省时，而且可以降低标准化失败率，但也有缺点，就是切割大图的做法太粗糙，另外保存大图会引来额外的空间开销。

#### predict_in_patch.py
处理的逻辑和label_get_patch中用到的方法差不多，在这里分别展示了使用keras和pytorch在大图中预测再将预测结果合并矩阵(以便于后续保存)的方法。


### ./tricks_in_processing_and_training/
请仔细阅读**./tricks_in_processing_and_training/README.md**文档,里面详尽介绍了进行数据处理以及模型训练时必须注意的一些陷阱/技巧。


### ./normalization/
里面存放颜色标准化工具**StainTools**的核心方法，关于**StainTools**的使用说明，详见https://github.com/Peter554/StainTools
在这里要补充一个说明，V方法虽然比M方法更先进，但是使用V方法遇上低对比度的图像时很容易会让系统产生``core dump！``的浮点计数报错，该报错原因处在操作系统的c++相关库上，无法被python自带的异常捕捉机制处理。为了尽可能避免这种意外的中断,可以进行条件判断才进行颜色转换操作,见下面**stain_trans.py**使用范例中的的`slide_region[np.std(slide_region,axis=2)<3]`。经过我多次对比,基于对比度作为判断全黑/全灰比直接指定RGB通道像素值范围更合适。使用M方法遇上低对比度的图片时也会转换失败，但可以使用python的异常捕捉机制捕捉这个异常。


### ./spams-2.6.1/
使用上述的颜色标准化工具必须要安装2.6.1的**spams**，但是直接从公开镜像使用``pip install ``安装很可能会出现失败，此时可以直接运行里面的setup.py文件进行安装(``python setup.py install``)。


### ./utils/
这里集成了我在项目中常用的方法封装。

#### opencv_utils.py
将我常用到的opencv方法封装成一个类。

#### openslide_utils.py
将我常用到的openslide_utils方法封装成一个类。

#### stain_trans.py
将颜色标准化的方法进行封装,使用方法如下,注意要加上异常捕捉模块。:
```python
from utils.stain_trans import standard_transfrom
from utils.openslide_utils import Slide

normalize_target_img = io.imread('TUM-AGQGDHKE.tif')
normalize_method = standard_transfrom(normalize_target_img,'V')
slide = Slide(svs_file)
slide_region = slide.read_region((w_cor,h_cor),0,(patch_size,patch_size)).convert('RGB')
slide_region = np.array(img)
if np.sum(slide_region[np.std(slide_region,axis=2)<3]) < slide_region.shape[0] * slide_region.shape[1] *0.4:
    try:
        slide_region = normalize_method.transform(slide_region)
    except Exception:
        print(basename + ' read_region failed in ' + str((w_cor,h_cor)))

```

#### tissue_utils
对亮度正常的病理WSI图像，进行背景/组织提取的方法(默认在2级下采样进行，提取出来的矩阵，0代表背景，1代表组织)。该方法会基于轮廓面积来过滤离散的组织区域。

#### xml_utils.py
根据xml格式的标注文件，提取标注轮廓的方法，不过本方法仅对示例标注文件(**16559_.xml**)的结构格式有效。使用方法如下:
```python
from utils.xml_utils import xml_to_region,region_handler
from utils.openslide_utils import Slide
import numpy as np

xml_file = os.path.join(INPUT_XML_DIR,pkl_name + '.xml')
slide = Slide(svs_file)
tile =slide.get_thumb()
if xml and os.path.exists(xml_file):
    region_list,region_class = xml_to_region(xml_file)
    svs_im_npy = region_handler(tile, region_list, region_class,slide.get_level_downsample())
    svs_im_npy = np.array(svs_im_npy.convert('RGBA'))
else:
    svs_im_npy = np.array(tile.convert('RGBA'))
```
