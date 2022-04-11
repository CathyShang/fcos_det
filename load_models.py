# 这是一个测试文件，加载训练好的模型 到底并行不并行
import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from model.fcos import FCOSDetector
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model = FCOSDetector(mode="inference")
model = torch.nn.DataParallel(model)
model = model.cuda().eval()

# model.load_state_dict(torch.load("/home/wangzy/shangzq/fcos_checkpoint/test_0/model_10.pth",map_location=torch.device('cpu')))
model.load_state_dict(torch.load("/home/wangzy/shangzq/fcos_checkpoint/04-03_23-42_fcos/model_160.pth",map_location=torch.device('cpu')))