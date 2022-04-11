# 任务
遥感图像目标检测

## 数据集

- NWPU VHR-10
    - 包含10类正例样本650张以及不包含给定对象类的任何目标的150张反例图像（背景），正例图像中至少包含1个实例，总共有3651个目标实例

- LEVIR
    - 数据集包含21952幅遥感图像和11028个实例
    - 图片尺寸为600*800
    - 数据集中的实例分为3个类别：飞机、船舶、储油罐，其中，飞机实例有4724个、轮船实例有3025个、储油罐实例有3279个

## 评价指标
- AP
- FLOPs



# 运行
使用COCO格式
## 训练
```
python train_coco.py
```


### 参考文献
1. G. Cheng, P. Zhou, and J. Han. Learning rotation-invariant convolutional neural networks for object detection in VHR optical remote sensing images. IEEE Trans. Geosci. Remote Sens., 54(12):7405–7415,2016.
2. Z. Zou, Z. Shi. Random Access Memories: A New Paradigm for Target Detection in High Resolution Aerial Remote Sensing Images. IEEE Transactions on Image Processing, 27(3), 1100–1111, 2018