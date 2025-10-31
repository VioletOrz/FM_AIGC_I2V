###########################################################################
#这个代码是用来映射角色别名和角色正式英文名称的
#yaml中每个id下面第一个名字是角色的中文名，第二个是官方英文名
#后面的则是别名
#如果你的文件夹的名字能和某个id下面的名字匹配，
#那么这个代码会把他复制到指定文件夹下，并重命名为官方英文名
###########################################################################
import yaml
import os
import shutil

def load_characters_yaml(characters_path):
    with open(characters_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config
        
    #return characters

def process_config_to_dict(config:list):
    character_dict = {}
    for config in config_list:
        for _, value in config.items():
            if len(value) == 1:
                true_name = value[0]
            elif len(value) == 0:
                continue
            else:
                true_name = value[1]
            for name in value:
                character_dict[name] = true_name
    return character_dict

def copy_and_rename_folders(source_folder, target_root, character_dict):
    # 遍历 source_folder 中的文件夹
    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        
        # 检查路径是否为文件夹
        if not os.path.isdir(folder_path):
            continue
        
        # 检查文件夹名称是否在字典中
        if folder_name in character_dict:
            # 构建目标文件夹路径，并使用字典的值进行重命名
            target_folder = os.path.join(target_root, character_dict[folder_name])
            os.makedirs(target_folder, exist_ok=True)
            
            # 复制文件夹到目标位置
            shutil.copytree(folder_path, target_folder, dirs_exist_ok=True)
            print(f"已将文件夹 '{folder_name}' 复制并重命名为 '{character_dict[folder_name]}' 到 {target_folder}")
        else:
            print('#' * 50)
            print(f"文件夹 '{folder_name}' 未在字典中找到，跳过")
            print('#' * 50)

if __name__ == "__main__":

    sr_config_path = "./config/sr.yaml"
    zzz_config_path = "./config/zzz.yaml"
    gs_config_path = "./config/genshin.yaml"

    sr_config = load_characters_yaml(sr_config_path)
    zzz_config = load_characters_yaml(zzz_config_path)
    gs_config = load_characters_yaml(gs_config_path)

    config_list = [sr_config, zzz_config, gs_config]

    character_dict = process_config_to_dict(config_list)

    source_folder = r'C:\Users\Violet\Desktop\AItalk表情包'
    target_root = r"E:/图包/表情包01"

    copy_and_rename_folders(source_folder, target_root, character_dict)

