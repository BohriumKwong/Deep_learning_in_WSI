# 进行数据处理以及模型训练时必须注意的一些陷阱/技巧

##  涉及的package

``PIL``
``matplotlib.pyplot``
``opencv-python``
``openslide-python``
``tensorflow``
``Keras``
``torch``



## 大图对象操作
一般来说python可支持操作的图像类型的对象大小有上限的，这也是为什么自带的图像处理工具库**PIL**无法读取无法读取WSI(Whole Slide Image)格式的图片的原因，但如果仍坚持直接用**PIL**处理这些大图像对象(通常我们不建议这样做),可通过一下语句进行修改，突破默认的限制：

```python
from PIL import Image
Image.MAX_IMAGE_PIXELS=500000000 #这里可以设一个很大的数值
```

## 在IPython中画出大图
针对Spyder或者Jupyter Notebook这些集成IPython窗口的编译环境，如果希望用matplotlib能对一些大图(几千像素×几千像素级别)以比较友好的方式画出来时，需要手工指定画布大小，否则会按默认大小来显示，以下是原效果：
![](small_plot.png)
可通过以下语句重新指定画布大小：

```python
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15, 15
```
显示的效果如下：
![](big_plot.png)

## opencv-python 默认通道顺序
如果在模型训练前后的预处理过程中需要用到opencv来处理图像，必须要谨记的一点时，opencv默认读写图像的通道顺序并非是RGB而是**BGR**！！！！
### 读取图片
一般情况下，为了避免后续操作混乱，如果是用于模型训练/预测读取图片，尽量避免使用opencv进行图片读取，因为其他框架/包处理图片时默认都是RGB顺序，如：
```python
import cv2
cv_test_read = cv2.imread('16558.png')
plt.imshow(cv_test_read)
```
显示的效果如下（注意和上图进行对比）：
![](bgr_16588.png)
为此，需要在使用opencv读取图片后，马上进行转换处理：
```python
import cv2
img = cv2.imread('16558.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#####如果确定更不需要使用opencv对图片进行处理，建议直接用skimage
import skimage.io as io
img = io.imread('16558.png')
```
### 保存图片
即使你的图片是RGB通道顺序的，但如果有使用opencv对图像进行修改(resize这些形态操作除外)，如画框，修改等，就必须要要以opencv进行图片的保存，但需要注意的是，只要是使用opencv进行图片保存，都必须在处理前将图片转为BGR通道顺序，否则保存下来的图片并非是RGB下真实的颜色:
```python
import cv2
import skimage.io as io
img_new = cv2.rectangle(img, (int(nuclei_info[l,0]),int(nuclei_info[l,1])),(int(nuclei_info[l,0]+1),int(nuclei_info[l,1]+1)),color_flag_25,3)
cv2.imwrite(save_dir,cv2.cvtColor(img_new, cv2.COLOR_RGB2BGR))
#####如果确定更不需要使用opencv对图片进行处理，建议直接用skimage进行图片保存(前提是通道顺序必须是RGB)
io.imsave(save_dir,img)
```
### 使用深度学习框架进行预测
由于tensorflow/pytorch读取图像都是默认为RGB通道顺序的，在进行预测的时候，必须要要确定图片不是BGR顺序的，否则预测的结果会相差很远。


## openslide的trick说明

### 兼容格式
实际上openslide很多使用习惯和python自带的**PIL**相同，首先不管是``slide.get_thumbnail(level)``方法还是``slide.read_region((w_coordinate, h_coordinate), level, (patch_w, patch_h))``方法，返回的对象都是Image对象而非numpy array对象。因此为便于对返回的Image对象进行后续处理，一般都会将其转换为数组。由于Image对象，所以默认的通道顺序自然也是RGB，只需要直接使用``np.array()``方法则可:
``region_img_arr =  np.array(slide.read_region((w_coordinate, h_coordinate), level, (patch_w, patch_h)))``
但需要注意的是，直接转换为数组时候，其通道数的深度并非为3而是4，因为获取的数组，实际上是包含了alpha通道(透明度)，即转换后的数组，第三维度依次是**[R,G,B,A]**。一般情况下这个WSI图像不存在透明度处理的问题，因此可以将这个通道直接丢弃，只取前三个通道，保证进行np.array转换后能兼容常规的图像处理工具格式则可，如：
``region_img_arr =  np.array(slide.read_region((w_coordinate, h_coordinate), level, (patch_w, patch_h)))[:,:,:3]``


### 坐标顺序
之后是坐标顺序，
实际上openslide很多使用习惯和python自带的**PIL**相同，首先不管是``slide.get_thumbnail(level)``方法还是``slide.read_region((w_coordinate, h_coordinate), level, (patch_w, patch_h))``方法又或者是``slide.level_dimensions[level]``，牵涉到坐标的设置，都是**(图像的宽，图像的高)**，和array中矩阵的顺序(图像的高，图像的宽)刚好相反，如：
![](coor.png)
因此，上述所有参数都是先width, 之后才是height，而基于numpy array的skimage、cv2甚至是scipy这些包，处理图像(矩阵)时，都是先height, 之后才是width，如
``img[height,width]``



## 使用深度学习框架加载ImageNet权重进行相关操作时
如果当前的场景需要加载ImageNet预训练权重，需要注意的时候，必须同步相关的数据预处理方法，详见：
https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/python/keras/applications/imagenet_utils.py
```python
def _preprocess_symbolic_input(x, data_format, mode):
  """Preprocesses a tensor encoding a batch of images.
  Arguments:
      x: Input tensor, 3D or 4D.
      data_format: Data format of the image tensor.
      mode: One of "caffe", "tf" or "torch".
          - caffe: will convert the images from RGB to BGR,
              then will zero-center each color channel with
              respect to the ImageNet dataset,
              without scaling.
          - tf: will scale pixels between -1 and 1,
              sample-wise.
          - torch: will scale pixels between 0 and 1 and then
              will normalize each channel with respect to the
              ImageNet dataset.
  Returns:
      Preprocessed tensor.
  """
  global _IMAGENET_MEAN

  if mode == 'tf':
    x /= 127.5
    x -= 1.
    return x

  if mode == 'torch':
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  else:
    if data_format == 'channels_first':
      # 'RGB'->'BGR'
      if K.ndim(x) == 3:
        x = x[::-1, ...]
      else:
        x = x[:, ::-1, ...]
    else:
      # 'RGB'->'BGR'
      x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None

  if _IMAGENET_MEAN is None:
    _IMAGENET_MEAN = constant_op.constant(-np.array(mean), dtype=K.floatx())

  # Zero-center by mean pixel
  if K.dtype(x) != K.dtype(_IMAGENET_MEAN):
    x = K.bias_add(x, math_ops.cast(_IMAGENET_MEAN, K.dtype(x)), data_format)
  else:
    x = K.bias_add(x, _IMAGENET_MEAN, data_format)
  if std is not None:
    x /= std
  return x
```
以Keras为例，仅在训练的时候在 ImageDataGenerator指定rescale=1./255是不够的，实际上还需要自己额外进行去均值和标准差。ImageNet数据集的均值和标准差如下:
```
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```
以Keras为例，可以在ImageDataGenerator的引用中增设preprocessing_function的定义，如：
```python
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

def imagenet_processing(image):
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
    for i in range(3):
    	image[:,:,i] -= mean[i]
        image[:,:,i] /= std[i]
    return image

train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip = True,
        vertical_flip = True,
        preprocessing_function = imagenet_processing
        )
train_generator = train_datagen.flow_from_directory(
        'train_dataset_dir',  # 训练数据路径
        target_size=(input_size_h, input_size_w),  # 设置图片大小
        batch_size=batch_size # 批次大小
        )

```

以pytorch为例，训练时可以在使用数据生成器的时候，增加torchvision.transforms的Normalize方法的调用，如:
```python
from torchvision import datasets, transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def get_dataloader():
    data_transforms = {
    'train':transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ]),
    'val':transforms.Compose([ transforms.ToTensor(),transforms.Normalize(mean,std)
    ]),}

    image_datasets = {x: datasets.ImageFolder(self.datapath[x],data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = self.batch_size, shuffle = True, num_workers=0)  for x in ['train', 'val']}
```



需要注意的是上述提及到的ImageNet权重数据预处理的方法和框架无关，只要是加载框架对应的模型的ImageNet预训练权重，都必须进行同样的预处理，只有这样才能真正发挥ImageNet预训练权重的作用。用于预测时同理。