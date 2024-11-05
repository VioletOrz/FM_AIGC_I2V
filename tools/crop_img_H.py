import os
from PIL import Image

def crop_images_in_subfolders(folder_path, crop_height=100):
    # 遍历所有子目录和文件
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".webp"):
                image_path = os.path.join(root, file_name)
                
                # 打开图像
                image = Image.open(image_path)
                width, height = image.size

                # 计算裁剪后的区域（上下各裁剪 crop_height 像素）
                cropped_image = image.crop((0, crop_height, width, height - crop_height))

                # 保存裁剪后的图像，覆盖原文件
                cropped_image.save(image_path)
                print(f"Cropped {file_name} in {root}")

# 使用指定的文件夹路径
folder_path = r"C:\Users\Violet\Desktop\Difference_S"
crop_images_in_subfolders(folder_path, crop_height=100)
