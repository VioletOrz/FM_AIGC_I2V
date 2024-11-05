
import os
import shutil

def copy_specified_folders_with_structure(src_folder, target_folder, folder_names):
    """
    在src_folder的所有子目录中查找指定名称的文件夹，将其下的所有文件和子文件夹复制到目标目录target_folder，
    并保留其父目录路径结构。
    
    参数:
    - src_folder: 源目录
    - target_folder: 目标目录
    - folder_names: 要查找的文件夹名称列表
    """
    for root, dirs, files in os.walk(src_folder):
        for dir_name in dirs:
            # 如果当前文件夹名称在指定名称列表中
            if dir_name in folder_names:
                source_dir_path = os.path.join(root, dir_name)
                relative_path = os.path.relpath(root, src_folder)
                
                # 在目标文件夹创建保留父目录结构的路径
                dest_dir_path = os.path.join(target_folder, relative_path, dir_name)
                
                # 递归复制整个文件夹内容到目标路径
                shutil.copytree(source_dir_path, dest_dir_path, dirs_exist_ok=True)
                print(f"Copied {source_dir_path} to {dest_dir_path}")

# 使用示例
src_folder = r"E:\图包\004"
target_folder = r"E:\图包\004_i"
folder_names = ["Difference_I","cropped_dir"]  # 可以指定多个文件夹名称
copy_specified_folders_with_structure(src_folder, target_folder, folder_names)



