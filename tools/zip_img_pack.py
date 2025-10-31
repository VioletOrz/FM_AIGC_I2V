########################################################################
#批量压缩指定文件夹下的子文件夹
#给zip添加父文件夹
########################################################################

import os
import shutil
import zipfile
import os
import shutil
import zipfile

def compress_all_folders(src_folder):
    """
    压缩指定目录下的所有文件夹，命名为 文件夹的名字.zip。

    :param src_folder: 根目录路径
    
    """

    subfolders = [
        os.path.join(src_folder, name)
        for name in os.listdir(src_folder)
        if os.path.isdir(os.path.join(src_folder, name))
    ]

    for folder_path in subfolders:
        if not os.path.exists(folder_path):
            print(f"文件夹 {folder_path} 不存在")
            return
        
        # 确定目标 ZIP 文件路径
        zip_path = f"{folder_path}.zip"
        
        try:
            # 压缩文件夹
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 将文件相对于文件夹的路径添加到 ZIP 中
                        arcname = os.path.relpath(file_path, folder_path)
                        zipf.write(file_path, arcname)
            print(f"文件夹已成功压缩为: {zip_path}")
            
            # 删除原文件夹
            shutil.rmtree(folder_path)
            print(f"原文件夹已删除: {folder_path}")
        except Exception as e:
            print(f"操作失败: {e}")


def move_zip_to_parent_folder(src_folder):
    """
    将指定目录下的所有 .zip 文件添加一个父目录，即 a.zip 变为 a/a.zip。
    
    :param src_folder: 根目录路径
    """
    for item in os.listdir(src_folder):
        item_path = os.path.join(src_folder, item)
        if os.path.isfile(item_path) and item.lower().endswith(".zip"):
            # 创建以 ZIP 文件名为名称的目录
            zip_name = os.path.splitext(item)[0]  # 去掉 .zip 后缀
            new_dir = os.path.join(src_folder, zip_name)
            os.makedirs(new_dir, exist_ok=True)
            
            # 移动 ZIP 文件到新目录中
            new_zip_path = os.path.join(new_dir, item)
            shutil.move(item_path, new_zip_path)
            print(f"Moved {item_path} to {new_zip_path}")


def organize_zip_files(base_dir):
    """
    遍历指定目录下的所有 .zip 文件，为每个文件创建一个父目录，
    将 .zip 文件移动到父目录中。
    对于 "_trans" 结尾的 .zip 文件，移动到去掉 "_trans" 后同名的父目录。

    参数:
    - base_dir: 指定的根目录
    """
    if not os.path.exists(base_dir):
        print(f"目录 {base_dir} 不存在")
        return

    # 遍历根目录及其子目录的所有 .zip 文件
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.zip'):
                file_path = os.path.join(root, file)

                # 判断是否是 "_trans" 结尾的文件
                if file.endswith('_trans.zip'):
                    # 去掉 "_trans" 的父目录名
                    parent_dir_name = file[:-10]  # 去掉 "_trans.zip"
                else:
                    # 直接使用文件名（去掉 .zip 后缀）
                    parent_dir_name = file[:-4]  # 去掉 ".zip"

                # 目标父目录路径
                parent_dir_path = os.path.join(base_dir, parent_dir_name)

                # 如果父目录不存在，则创建
                os.makedirs(parent_dir_path, exist_ok=True)

                # 移动文件到父目录
                shutil.move(file_path, os.path.join(parent_dir_path, file))
                print(f"Moved: {file_path} -> {os.path.join(parent_dir_path, file)}")

# 示例使用
src_folder = r"E:\图包006_r"  # 根目录
compress_all_folders(src_folder)
#move_zip_to_parent_folder(src_folder)
organize_zip_files(src_folder)

#src_folder = r"E:\图包006_i"  # 根目录
