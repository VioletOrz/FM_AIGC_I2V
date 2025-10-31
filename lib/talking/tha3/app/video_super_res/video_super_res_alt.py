############################################
##该脚本用来处理文件夹中的图像，对图像进行超分辨率处理

import cv2
import torch
import os
import sys
import torchvision
sys.path.append('D:\\FaceMind_AIGC\\APISR')
from test_code import inference
from test_code.test_utils import load_grl, load_rrdb, load_dat, load_cunet

def super_resolve_video(weight_path, input_img_path, output_img_path):
    # 加载模型权重
    if weight_path=="D:\\FaceMind_AIGC\\APISR\\pretrained\\2x_APISR_RRDB_GAN_generator.pth":
        generator = inference.load_rrdb(weight_path, scale=2)
    elif weight_path=="D:\\FaceMind_AIGC\\APISR\\pretrained\\4x_APISR_RRDB_GAN_generator.pth":
        generator=inference.load_rrdb(weight_path, scale=4)
    else:
        generator=inference.load_grl(weight_path, scale=4)
    generator = generator.to(torch.float32)

    for file in os.listdir(input_img_path):
        img=cv2.imread(os.path.join(input_img_path, file))
        img_height=img.shape[0]
        img_width=img.shape[1]
        new_img=img
        if(img_height*img_width>=2000*2000):
            new_img=cv2.resize(img, (img_width//2,img_height//2))
            # 对图片进行超分辨率处理
        sr_img=inference.super_resolve_img_without_path(generator,new_img, weight_dtype=torch.float32, downsample_threshold=-1, crop_for_4x=False)


        if not os.path.exists(output_img_path):
            os.makedirs(output_img_path)
        torchvision.utils.save_image(sr_img, os.path.join(output_img_path, file))

# mode=['a','delta','e','i','o','u','raise','lower','smirk']
# strength=[1,0.75,0.5,0.25]
# for m in mode:
#     for s in strength:
#         mydir=m+'_'+str(s)
#         super_resolve_video(r"D:\FaceMind_AIGC\APISR\pretrained\4x_APISR_GRL_GAN_generator.pth", "C:\\Users\\FM\\Desktop\\facial\\annoyed\\"+mydir,"C:\\Users\\FM\\Desktop\\facial\\sr\\annoyed\\"+mydir)