import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_difference_mask(image_path1, image_path2, img_num, mode, strength):
    # Read images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate difference
    difference = cv2.absdiff(gray1, gray2)

    # Thresholding
    _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Create mask with alpha channel
    mask = np.zeros((image1.shape[0], image1.shape[1], 4), dtype=np.uint8)

    # Set RGB values where there is difference
    mask[dilated == 255, :3] = image2[dilated == 255, :]

    # Set alpha channel to 255 (fully opaque) where there is difference
    mask[dilated == 255, 3] = 255

    # Save the mask as PNG with alpha channel
    mask_path = f'C:/Users/Violet/Desktop/facial/chayi2/{mode}_{strength}/img_{img_num + 1}.webp'
 
    cv2.imwrite(mask_path, mask)

    return mask_path

transform_mode=['a','i','o']
transform_strength=[1.0,0.5]
# transform_mode=['a']
# transform_strength=[1]

for mode in transform_mode:
    print(f"###################################################开始合成{mode}表情###################################################")
    for strength in transform_strength:
        if  mode=='i' or mode=='o':
             strength=1.0
        print(f"###################################################开始合成{mode}表情的强度为{strength}###################################################")
        if not os.path.exists(f'C:/Users/Violet/Desktop/facial/chayi2/{mode}_{strength}'):
         os.makedirs(f'C:/Users/Violet/Desktop/facial/chayi2/{mode}_{strength}')
        images_dir1 = r"C:/Users/Violet/Desktop/facial/change_menglong/b_0.0"
        #images_dir2 = r"C:/Users/Violet/Desktop/facial\change_pic"
        files1 = os.listdir(images_dir1)
        #files2 = os.listdir(images_dir2)
        for i in tqdm(range(len(files1)), desc="生成差异图"):
            image_path1 = f'C:/Users/Violet/Desktop/facial/change_size/b_0.0/img_{i + 1}.webp'
            image_path2 = f'C:/Users/Violet/Desktop/facial/change_size/{mode}_{strength}/img_{i + 1}.webp'
            print(image_path1)
            print(image_path2)
            generate_difference_mask(image_path1, image_path2, i, mode, strength)


import shutil
import os

def copy_folder_to_another(src_folder, dest_folder):
    try:
        # 检查目标文件夹是否存在，不存在则创建
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # 获取源文件夹的名称
        folder_name = os.path.basename(src_folder.rstrip('/\\'))
        
        # 目标文件夹路径，包含源文件夹的名称
        target_path = os.path.join(dest_folder, folder_name)
        
        # 使用 shutil.copytree 方法将源文件夹复制到目标文件夹中
        shutil.copytree(src_folder, target_path)
        print(f"文件夹已成功复制到：{target_path}")
    except Exception as e:
        print(f"复制文件夹时出错: {e}")


# 使用示例
src_folder = r"C:/Users/Violet/Desktop/facial\change_size\b_0.0" # 替换为源文件夹的路径
dest_folder = r"C:/Users/Violet/Desktop/facial\chayi2"  # 替换为目标文件夹的路径

copy_folder_to_another(src_folder, dest_folder)

