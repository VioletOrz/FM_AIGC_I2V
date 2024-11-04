from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm
def resize_image(input_path, output_path, scale_factor=1):
    # 打开原始图像
    img = Image.open(input_path)
    
    # 计算新的尺寸
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    
    # 使用 LANCZOS 滤波器调整图像尺寸
    resized_img = img.resize(new_size, Image.LANCZOS)
    
    # 保存调整后的图像
    resized_img.save(output_path)

def resize_image_ref(img_path, ref_img_path,output_path):
    # 打开原始图像
    img = Image.open(img_path)
    ref_img = Image.open(ref_img_path)
    
    # 计算新的尺寸
    new_size = (int(ref_img.width), int(ref_img.height))
    
    # 使用 LANCZOS 滤波器调整图像尺寸
    resized_img = img.resize(new_size, Image.LANCZOS)
    
    # 保存调整后的图像
    resized_img.save(output_path)

# image_path = r"C:/Users/Violet/Desktop/facial\change_chaofen\b_0.0\img_1_out.webp"
# output_path = r"C:/Users/Violet/Desktop/facial\change_chaofen\b_0.0\img_1_resized.webp"
# resize_image(image_path, output_path, scale_factor=1)

def resize_batch(input_file, output_file):

    transform_mode=['b','a','i','o']
    transform_strength=[1.0,0.5]
    # transform_mode=['a']
    # transform_strength=[1]

    folder_path = input_file+'/b_0.0'
    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    for mode in transform_mode:
        print(f"###################################################开始合成{mode}表情###################################################")
        if mode=='b':
            strength=0.0
            print(f"###################################################开始合成{mode}表情的强度为{strength}###################################################")
            if not os.path.exists(output_file+f'/{mode}_{strength}'):
                os.makedirs(output_file+f'/{mode}_{strength}')
            
            #files2 = os.listdir(images_dir2)
            for i in tqdm(range(file_count), desc="更改图片大小"):
                image_path = input_file+f'/{mode}_{strength}/img_{i + 1:03}.webp'
                output_path = output_file+f'/{mode}_{strength}/img_{i + 1:03}.webp'
                resize_image(image_path, output_path, scale_factor=0.5)
                print(image_path)
        else:

            for strength in transform_strength:
                if  mode=='i' or mode=='o':
                    strength=1.0
                print(f"###################################################开始合成{mode}表情的强度为{strength}###################################################")
                if not os.path.exists(output_file+f'/{mode}_{strength}'):
                    os.makedirs(output_file+f'/{mode}_{strength}')
                    
                for i in tqdm(range(file_count), desc="更改图片大小"):
                    image_path = input_file+f'/{mode}_{strength}/img_{i + 1:03}.webp'
                    output_path = output_file+f'/{mode}_{strength}/img_{i + 1:03}.webp'
                    resize_image(image_path, output_path, scale_factor=0.5)
                    print(image_path)
# if not os.path.exists(f'C:/Users/Violet/Desktop/facial/source_picture-5'):
#                  os.makedirs(f'C:/Users/Violet/Desktop/facial/source_picture-5')
# for i in tqdm(range(1890), desc="更改图片大小"):
#             image_path = f'C:/Users/Violet/Desktop/facial/source_picture-4/img_{i + 1}.png'
#             output_path = f'C:/Users/Violet/Desktop/facial/source_picture-5/img_{i + 1}.webp'
#             resize_image(image_path, output_path, scale_factor=1)
#             print(image_path)









