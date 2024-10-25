import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_difference_mask(image_path1, image_path2, img_num, mode, strength, output_file):
    # Read images
    image1 = cv2.imread(image_path1, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)

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
    mask[dilated == 255, :4] = image2[dilated == 255, :]

    # Set alpha channel to 255 (fully opaque) where there is difference
    #mask[dilated == 255, 3] = 255

    # Save the mask as PNG with alpha channel
    mask_path = output_file+f'/{mode}_{strength}/img_{img_num + 1:03}.webp'
 
    cv2.imwrite(mask_path, mask)

    return mask_path

def diff_image(input_file, output_file):
    transform_mode=['a','i','o']
    transform_strength=[1.0,0.5]
    # transform_mode=['a']
    # transform_strength=[1]

    folder_path = input_file+'/b_0.0'
    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    for mode in transform_mode:
        print(f"###################################################开始合成{mode}表情###################################################")
        for strength in transform_strength:
            if  mode=='i' or mode=='o':
                strength=1.0
            print(f"###################################################开始合成{mode}表情的强度为{strength}###################################################")
            if not os.path.exists(output_file+f'/{mode}_{strength}'):
                os.makedirs(output_file+f'/{mode}_{strength}')
    
            for i in tqdm(range(file_count), desc="生成差异图"):
                image_path1 = input_file+f'/b_0.0/img_{i + 1:03}.webp'
                image_path2 = input_file+f'/{mode}_{strength}/img_{i + 1:03}.webp'
                print(image_path1)
                print(image_path2)
                generate_difference_mask(image_path1, image_path2, i, mode, strength, output_file)

    # 使用示例
    src_folder = input_file+"/b_0.0" # 替换为源文件夹的路径
    dest_folder = output_file  # 替换为目标文件夹的路径

    copy_folder_to_another(src_folder, dest_folder)



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



