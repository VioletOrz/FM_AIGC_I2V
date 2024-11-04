############################################
##该脚本用来处对单个图片进行超分辨率处理

import cv2
import torch
import os
import sys
import torchvision
sys.path.append('D:\\FaceMind_AIGC\\APISR')
from test_code import inference
from test_code.test_utils import load_grl, load_rrdb, load_dat, load_cunet

#第一个参数用来选择超分模型,第二个参数要求输入的是numpy格式的图片
def super_resolve_video(weight_path, img):
    # 加载模型权重
    if weight_path=="D:\\FaceMind_AIGC\\APISR\\pretrained\\2x_APISR_RRDB_GAN_generator.pth":
        generator = inference.load_rrdb(weight_path, scale=2)
    elif weight_path=="D:\\FaceMind_AIGC\\APISR\\pretrained\\4x_APISR_RRDB_GAN_generator.pth":
        generator=inference.load_rrdb(weight_path, scale=4)
    else:
        generator=inference.load_grl(weight_path, scale=4)
    generator = generator.to(torch.float32)

    img_height=img.shape[0]
    img_width=img.shape[1]
    new_img=img
    if(img_height*img_width>=1000*1000):
        new_img=cv2.resize(img, (img_width//2,img_height//2))
        # 对图片进行超分辨率处理
    sr_img=inference.super_resolve_img_without_path(generator,new_img, weight_dtype=torch.float32, downsample_threshold=-1, crop_for_4x=False)
    return sr_img

