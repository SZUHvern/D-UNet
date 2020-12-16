# D-UNet
D-UNet: a dimension-fusion U shape network for chronic stroke lesion segmentation
![D-Unet Architecture](D-Unet.png) 
# Author
Weijian Huang
# 项目简介
## 1. 功能
代码实现的功能
## 2. 性能
代码的性能，比如资源占用，准确率等。
## 3. 评估指标
比如mAP等。
## 4. 使用数据集
包括数据集名称、来源。如果不使用数据集，则留空。

# 运行环境与依赖
代码运行的环境与依赖。如下所示：

|类别|名称|版本|
|-----|-----|-----|
|os|ubuntu|16.04|
|深度学习框架|pytorch|0.4.0|
||opencv|3.4.9|

# 输入与输出
代码的输入与输出。如下所示：

|名称|说明|
|-----|-----|
|输入|RGB图像。大小为224X224（宽x高）|
|输出|标签。0表示背景，1表示人|

# 运行方式
在terminal下运行以下命令。
```shell
cd project_dir
python .\main.py --arg1 arg1 --arg2 arg2
```
