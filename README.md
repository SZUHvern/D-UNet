# D-UNet
D-UNet: a dimension-fusion U shape network for chronic stroke lesion segmentation
![D-Unet Architecture](D-Unet.png) 
# Author
Yongjin Zhou, Weijian Huang, Pei Dong, Yong Xia, and Shanshan Wang.
# 项目简介
## 1. 功能
采用D-UNet实现对ATLAS数据集的图像分割，兼顾了3D特征提取及高效的实现。
## 2. 性能
|DSC|Recall|Precision| Total parameters|
|-----|-----|-----|-----|
|0.5349±0.2763|0.5243±0.2910|0.6331±0.2958|8,640,163|
## 3. 评估指标
DSC(Dice Similarity Coefficient)、Recall、Precision
## 4. 使用数据集
ATLAS（Anatomical Tracings of Lesions-After-Stroke dataset）
[1] Liew, Sook-Lei, et al. "A large, open source dataset of stroke anatomical brain images and manual lesion segmentations." Scientific data 5 (2018): 180011.
# 运行环境与依赖
代码运行的环境与依赖。如下所示：

名称|版本|
|-----|-----|
|ubuntu|16.04|
|Tensorflow|1.10.0|
|Keras|2.2.0|
|Python|3.6.0|
# 输入与输出
代码的输入与输出。如下所示：

|名称|说明|
|-----|-----|
|输入|3D灰度图像，大小为192X192X4X1（宽x高x切片数x通道）的连续切片。
|输出|分割结果。0表示背景，1表示脑卒中区域|

# 运行方式
python Stroke_segment.py
