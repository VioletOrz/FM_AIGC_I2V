##########################################################################
#这段代码是用于复制一个完整图包中某些特定图包的，这样就不用手动一个个的去复制了
#复制过程中会保留除根目录外的父目录结构
#如果你不想保存某些父目录结构，可以使用copy_specified_folders_with_structure_deep函数
#retain_depth参数是指定从根目录开始，不保留几级目录结构
#########################################################################
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

def copy_specified_folders_with_structure_deep(src_folder, target_folder, folder_names, retain_depth=None):
    """
    在src_folder的所有子目录中查找指定名称的文件夹，将其下的所有文件和子文件夹复制到目标目录target_folder，
    并保留其父目录路径结构，且可以指定保留的目录层级深度。
    
    参数:
    - src_folder: 源目录
    - target_folder: 目标目录
    - folder_names: 要查找的文件夹名称列表
    - retain_depth: 要保留的目录层级深度（整数，None表示保留完整路径）
    """
    for root, dirs, files in os.walk(src_folder):
        for dir_name in dirs:
            # 如果当前文件夹名称在指定名称列表中
            if dir_name in folder_names:
                source_dir_path = os.path.join(root, dir_name)
                relative_path = os.path.relpath(root, src_folder)
                
                # 如果指定了保留深度，截取相应层级的路径
                if retain_depth is not None:
                    parts = relative_path.split(os.sep)
                    # 保留从相对路径的某个深度开始的部分
                    relative_path = os.sep.join(parts[-retain_depth:])
                
                # 在目标文件夹创建保留父目录结构的路径
                dest_dir_path = os.path.join(target_folder, relative_path, dir_name)
                 
                # 递归复制整个文件夹内容到目标路径
                shutil.copytree(source_dir_path, dest_dir_path, dirs_exist_ok=True)
                print(f"Copied {source_dir_path} to {dest_dir_path}")

# 使用示例
if __name__ == "__main__":
    src_folder = r"E:\图包\005"
    target_folder = r"E:\图包005_test"
    folder_names = ["Difference_I","cropped_dir"]  # 可以指定多个文件夹名称
    retain_depth = 1 #保留父目录时跳过几级目录结构 例如 已知目录a/b/c/d，src_folder = a，folder_names = ['d']，如果 retain_depth = 0 那么保留的目录结构 b/c/d ，retain_depth = 1 那么保留的目录结构 c/d，依次类推
    copy_specified_folders_with_structure_deep(src_folder, target_folder, folder_names, retain_depth)


