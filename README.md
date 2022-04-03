# 任务
自动驾驶任务中的无锚框目标检测，要求模型需要具有实时性

# 数据集
- KITTI (2D/3D)
    [[download](http://www.cvlibs.net/datasets/kitti/) 
    / [paper](https://arxiv.org/abs/1803.09719)] (2011)
    - 3D物体检测类别：汽车、货车、卡车、行人、自行车、电车、其他
- SODA10M (2D)
    [[download](https://soda-2d.github.io/download.html)
    / [paper](https://arxiv.org/pdf/2106.11118.pdf)] (2021)
    - 2万张有标签图片`5.6G`, 1000万张无标签图片`2T`
    - 6种类别：行人、自行车、汽车、卡车、电车、三轮车
    - 地平线保持在图像的中心，车内的遮挡不超过整个图像的15%
    - 提供者：华为诺亚方舟实验室&中山大学


# 评价指标
- AP
- FLOPs