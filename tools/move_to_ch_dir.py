######################################################################################
#move_matching_folders
#这个代码是用来把生成好的角色图包统一放到中文角色名对应的文件夹下面去的，
#比如生成好的Kamisato Ayaka和Kamisato Ayaka_trans，这两个图包是一个角色的，
#这个代码会在config\characters.yaml中寻找这两个包对应的中文角色名，然后将两个包一起放到
#中文文件夹下面去
#copy_files_to_directories这个代码则是用来把生图时使用的源图像统一打包复制到一起去的
#这两个函数是给Auto_queue提供的功能函数
######################################################################################
import os
import shutil
import yaml

import os
import shutil
import yaml

def load_yaml_data(yaml_path):
    # 读取 YAML 文件，加载列表和字典
    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data.get("image_path", []), data.get("character_name_dict", {})

def copy_files_to_directories(yaml_path, target_root):
    # 加载数据
    image_path, character_name_dict = load_yaml_data(yaml_path)
    
    # 在目标根目录下创建 ani-pic 文件夹
    ani_pic_folder = os.path.join(target_root, "ani-pic")
    os.makedirs(ani_pic_folder, exist_ok=True)

    # 遍历 image_path 和 character_name_dict
    for file, (key, folder_name) in zip(image_path, character_name_dict.items()):
        # 构建目标文件夹路径
        target_folder = os.path.join(target_root, folder_name)
        
        # 检查并创建目标文件夹
        os.makedirs(target_folder, exist_ok=True)

        # 复制文件到目标文件夹
        target_path = os.path.join(target_folder, os.path.basename(file))
        shutil.copy2(file, target_path)
        print(f"文件 {file} 已复制到 {target_path}")

        # 同时复制文件到 ani-pic 文件夹
        ani_pic_target = os.path.join(ani_pic_folder, os.path.basename(file))
        shutil.copy2(file, ani_pic_target)
        print(f"文件 {file} 同时复制到 {ani_pic_folder}")

def load_character_name_dict(yaml_path):
    # 读取 YAML 文件，加载字典
    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data.get("character_name_dict", {})

def print_character_name_dict(yaml_path):
    character_name_dict = load_character_name_dict(yaml_path)
    for english_name, chinese_name in character_name_dict.items():
        print(f"{english_name}: {chinese_name}")

def move_matching_folders(yaml_path, source_folder, target_root):
    # 加载字典
    character_name_dict = load_character_name_dict(yaml_path)
    d = {"a":"啊"}
    # 遍历 source_folder 中的所有子文件夹
    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        
        # 检查路径是否为文件夹
        if not os.path.isdir(folder_path):
            continue
        
        # 获取目标文件夹的中文名称
        target_name = None
        if folder_name in character_name_dict:
            target_name = character_name_dict[folder_name]
        elif folder_name.endswith("_trans"):
            base_name = folder_name.rsplit("_trans", 1)[0]
            if base_name in character_name_dict:
                target_name = character_name_dict[base_name]
        
        # 如果找到了目标名称，则移动文件夹
        if target_name:
            target_folder = os.path.join(target_root, target_name)
            os.makedirs(target_folder, exist_ok=True)
            shutil.move(folder_path, os.path.join(target_folder, folder_name))
            print(f"已将 {folder_name} 移动到 {target_folder}")
        else:
            print(f"跳过未匹配的文件夹：{folder_name}")

# 使用示例
if __name__ == "__main__":
    yaml_path = "./config/characters.yaml"
    source_folder = 'C:/Users/Violet/Desktop/facial/'
    target_root = "E:/图包/005"
    print_character_name_dict(yaml_path)
    move_matching_folders(yaml_path, source_folder, target_root)

    # 使用示例
    # 加载数据并执行文件复制
    copy_files_to_directories(yaml_path, target_root)