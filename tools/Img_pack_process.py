#################################################################################################################################################
#rename_webp_files：重命名 webp 文件，将 img_001 类似的命名方式改为 img_1。
#convert_webp_to_png：遍历目录，将所有 webp 文件转换为 png，并保存到相同目录下。
#delete_trans_folders：删除包含 _trans 的文件夹及其内容。
#move_difference_i_contents：找到所有名为 Difference_I 的文件夹，将其中内容移动到上一级目录，并删除 Difference_I 文件夹。
#rename_and_move_cropped_images：复制并重命名 cropped_image_large.png 和 cropped_image_small.png 文件，移动到父目录，并使用父文件夹名作为前缀命名。
#delete_cropped_dirs(folder_path) 删除指定文件夹下所有名为 'cropped_dir' 的子文件夹及其内容
#process_b_0_0_folders(folder_path) 遍历指定文件夹下的所有名为“b_0.0”的文件夹，按顺序命名的 img_001.webp 到 img_100.webp 图像文件，并合成 GIF。
#请将 folder_path 替换为你的目标文件夹路径，然后运行脚本。
#################################################################################################################################################

import os
import shutil
from PIL import Image
from pathlib import Path

def rename_webp_files(folder_path):
    """
    遍历指定文件夹下的所有文件，找到所有webp格式的文件并重命名，
    把img_001-img_100这种命名方式改为img_1-img_100。
    """
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.webp'):
                old_path = os.path.join(root, file_name)
                new_file_name = file_name.replace('img_', 'img_').lstrip("0") if file_name.startswith("img_0") else file_name
                new_path = os.path.join(root, new_file_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} to {new_path}")

def convert_webp_to_png(folder_path):
    """
    遍历指定文件夹下的所有文件，把所有的webp文件都转换并复制为png文件，
    存放到和webp文件相同的目录下。
    """
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.webp'):
                webp_path = os.path.join(root, file_name)
                png_path = os.path.join(root, file_name.replace('.webp', '.png'))
                with Image.open(webp_path) as img:
                    img.save(png_path, 'PNG')
                print(f"Converted: {webp_path} to {png_path}")

def delete_trans_folders(folder_path):
    """
    删除指定目录下所有名字中带 _trans 的文件夹及其中内容。
    """
    for root, dirs, _ in os.walk(folder_path):
        for dir_name in dirs:
            if '_trans' in dir_name:
                trans_folder_path = os.path.join(root, dir_name)
                shutil.rmtree(trans_folder_path)
                print(f"Deleted folder: {trans_folder_path}")

def move_difference_i_contents(folder_path, target_path):
    """
    遍历指定文件夹下的所有文件，找到所有名为“Difference_I”的文件夹，
    将其中的所有内容复制到“Difference_I”文件夹所在目录，并删除空的Difference_I文件夹。
    """
    for root, dirs, _ in os.walk(folder_path, target_path):
        if target_path in dirs:
            diff_i_path = os.path.join(root, target_path)
            for item in os.listdir(diff_i_path):
                src_path = os.path.join(diff_i_path, item)
                dest_path = os.path.join(root, item)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dest_path)
                print(f"Copied: {src_path} to {dest_path}")
            shutil.rmtree(diff_i_path)
            print(f"Deleted folder: {diff_i_path}")

def rename_and_move_cropped_images(folder_path):
    """
    遍历指定文件夹下的所有文件，找到所有名为“cropped_image_large.png”和“cropped_image_small.png”的文件，
    复制到父文件夹所在的目录，并分别重命名为“父文件夹名字-large”和“父文件夹名字-small”。
    """
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name in {"cropped_image_large.png", "cropped_image_small.png"}:
                parent_dir_name = Path(root).parent.name
                new_file_name = f"{parent_dir_name}-{'large' if 'large' in file_name else 'small'}.png"
                src_path = os.path.join(root, file_name)
                dest_path = os.path.join(os.path.dirname(root), new_file_name)
                shutil.copy2(src_path, dest_path)
                
                print(f"Copied and renamed: {src_path} to {dest_path}")
            #shutil.rmtree(root)

def delete_cropped_dirs(folder_path):
    """
    删除指定文件夹下所有名为 'cropped_dir' 的子文件夹及其内容。

    参数:
    - parent_folder: 要遍历的父文件夹路径。
    """
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            if dir_name == 'cropped_dir':
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)  # 删除文件夹及其中内容
                    print(f"已删除文件夹: {dir_path}")
                except Exception as e:
                    print(f"删除文件夹时出错: {e}")

def create_gif_from_images(img_folder, gif_output_path):
    """
    从指定文件夹中获取按顺序命名的 img_001.webp 到 img_100.webp 图像文件，并合成 GIF。
    
    参数:
    - img_folder: 图像文件所在文件夹路径
    - gif_output_path: GIF输出的路径
    """
    # 查找 img_001.webp 到 img_100.webp 的文件
    image_files = []
    for i in range(1, 101):
        img_path = os.path.join(img_folder, f"img_{i:03d}.webp")
        if os.path.exists(img_path):
            image_files.append(img_path)

    # 如果找到了图像文件，生成 GIF
    if image_files:
        # 打开图像文件
        images = [Image.open(img_path) for img_path in image_files]
        # 保存为 GIF，设置帧率为 30fps
        images[0].save(
            gif_output_path,
            save_all=True,
            append_images=images[1:],
            duration=1000/30,  # 30fps
            loop=0
        )
        print(f"GIF saved at {gif_output_path}")
    else:
        print(f"No valid images found in {img_folder}")

def process_b_0_0_folders(folder_path):
    """
    遍历指定文件夹下的所有名为“b_0.0”的文件夹，将其中的图像合成为 GIF 文件。
    
    参数:
    - base_folder: 根文件夹路径，遍历此文件夹及其子文件夹
    """
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            if dir_name == "b_0.0":
                b_0_0_folder = os.path.join(root, dir_name)
                parent_folder_name = os.path.basename(root)
                gif_output_path = os.path.join(root, f"{parent_folder_name}.gif")

                # 调用函数创建 GIF
                create_gif_from_images(b_0_0_folder, gif_output_path)

def main():
    folder_path = r'E:\图包005_i'


    # 删除带 "_trans" 的文件夹
    delete_trans_folders(folder_path)

    # 重命名 webp 文件
    #rename_webp_files(folder_path)

    # 转换 webp 文件为 png 文件
    #convert_webp_to_png(folder_path)

    # 移动 Difference_I 文件夹中的内容并删除文件夹
    target_path = "Difference_I"
    move_difference_i_contents(folder_path, target_path)

    # 重命名和移动 cropped_image_large 和 cropped_image_small 文件
    rename_and_move_cropped_images(folder_path)

    # 删除指定文件夹下所有名为 'cropped_dir' 的子文件夹及其内容
    delete_cropped_dirs(folder_path)

    # 创建 GIF 文件
    process_b_0_0_folders(folder_path)

if __name__ == "__main__":
    main()