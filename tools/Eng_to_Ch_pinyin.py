################################################################################################
#这个代码是将已经生成好并转移到中文目录下的以角色英文命名的图包改为以拼音命名
#或者是将已经生成好未转移到中文目录下的以角色英文命名的图包改为以拼音命名
#反过来也可以将拼音改为英文
#Auto_queue中 is_pinyin = False的情况下可以使用这个代码再次转到中文

#load_dict_from_yaml：加载 YAML 文件中的拼音到中文字典和英文到中文字典，并返回这两个字典。
#rename_folders_in_directory：遍历指定的根文件夹，根据 is_pinyin_to_english 参数来决定从拼音到英文重命名还是从英文到拼音重命名。支持处理 _trans 后缀。
#get_pinyin_from_english：从英文到拼音字典中获取拼音。
#get_english_from_pinyin：从拼音到英文字典中获取英文。
################################################################################################

import os
import yaml


def load_dict_from_yaml(yaml_file):
    """
    从YAML文件中加载字典。
    
    参数:
    - yaml_file: YAML文件路径
    
    返回:
    - 拼音到中文的字典和英文到中文的字典
    """
    with open(yaml_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    pinyin_dict = data.get("character_name_dict_pinyin", {})
    english_dict = data.get("character_name_dict_English", {})
    return pinyin_dict, english_dict

def rename_folders_in_directory(root_folder, is_pinyin_to_english, pinyin_dict, english_dict):
    """
    遍历指定目录，根据is_pinyin_to_english参数从英文转换为拼音或从拼音转换为英文。
    
    参数:
    - root_folder: 根文件夹路径
    - is_pinyin_to_english: 是否执行拼音到英文的重命名，如果为True，则从拼音重命名为英文，否则从英文重命名为拼音
    - pinyin_dict: 拼音到中文的字典
    - english_dict: 英文到中文的字典
    """
    for root, dirs, files in os.walk(root_folder):
        for dir_name in dirs:
            old_folder_path = os.path.join(root, dir_name)
            new_folder_name = ""

            # 处理拼音到英文的转换
            if is_pinyin_to_english:
                # 从拼音到英文的转换
                if "_trans" in dir_name:
                    base_name = dir_name.replace("_trans", "")
                    new_folder_name = get_english_from_pinyin(base_name, pinyin_dict, english_dict) + "_trans"
                else:
                    new_folder_name = get_english_from_pinyin(dir_name, pinyin_dict, english_dict)
            
            # 处理英文到拼音的转换
            else:
                # 从英文到拼音的转换
                if "_trans" in dir_name:
                    base_name = dir_name.replace("_trans", "")
                    new_folder_name = get_pinyin_from_english(base_name, english_dict, pinyin_dict) + "_trans"
                else:
                    new_folder_name = get_pinyin_from_english(dir_name, english_dict, pinyin_dict)
            
            new_folder_path = os.path.join(root, new_folder_name)

            # 进行重命名，如果新路径不存在且与旧路径不同
            if old_folder_path != new_folder_path and not path_exists_case_sensitive(new_folder_path):
                os.rename(old_folder_path, new_folder_path)
                print(f"Renamed: {old_folder_path} -> {new_folder_path}")
            else:
                print(f"Skipped renaming: {old_folder_path}, target already exists or no change needed.")

def get_english_from_pinyin(pinyin_name, pinyin_dict, english_dict):
    """
    根据拼音名称从拼音字典中查找对应的英文名。
    
    参数:
    - pinyin_name: 拼音名称
    - pinyin_dict: 拼音到中文的字典
    - english_dict: 英文到中文的字典
    
    返回:
    - 对应的英文名称
    """
    # 获取拼音对应的中文
    if pinyin_name in pinyin_dict:
        chinese_name = pinyin_dict.get(pinyin_name.lower(), pinyin_name)  # 使用小写拼音进行查找
        # 从中文查找英文名称，注意要进行精确匹配
        english_name = next((key for key, value in english_dict.items() if value == chinese_name), chinese_name)
        return english_name
    else:
        return pinyin_name

def get_pinyin_from_english(english_name, english_dict, pinyin_dict):
    """
    根据英文名称从英文字典中查找对应的拼音。
    
    参数:
    - english_name: 英文名称
    - english_dict: 英文到中文的字典
    - pinyin_dict: 拼音到中文的字典
    
    返回:
    - 对应的拼音名称
    """
    # 获取英文对应的中文
    if english_name in english_dict:
        chinese_name = english_dict.get(english_name, english_name)
        # 从拼音字典中查找对应的拼音
        pinyin_name = next((key for key, value in pinyin_dict.items() if value == chinese_name), chinese_name)
        return pinyin_name
    else:
        return english_name

def load_dict(yaml_file, is_pinyin):
    """
    从YAML文件中加载拼音到中文的字典并交换键值对。
    
    参数:
    - yaml_file: 存储拼音到中文的YAML文件路径
    
    返回:
    - 交换后的中文到拼音的字典
    """
    with open(yaml_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    if is_pinyin:
        pinyin_dict = data.get("character_name_dict_English", {})    
    else:
        pinyin_dict = data.get("character_name_dict_pinyin", {})

    
    # 交换字典中的键值对，将中文作为键，拼音作为值
    reversed_dict = {v: k for k, v in pinyin_dict.items()}
    return reversed_dict

def rename_subfolders_with_pinyin(root_folder, pinyin_dict):
    """
    遍历指定根文件夹下的所有中文文件夹，并将每个中文文件夹下的子文件夹重命名为中文父文件夹的拼音。
    如果子文件夹名称包含 '_trans'，则保留 '_trans' 后缀。
    
    参数:
    - root_folder: 根文件夹路径
    - pinyin_dict: 拼音字典，中文到拼音
    """
    for root, dirs, files in os.walk(root_folder):
        # 如果当前文件夹是根文件夹，进入其子文件夹
        if root == root_folder:
            for dir_name in dirs:
                # 获取当前中文父文件夹路径
                chinese_folder_path = os.path.join(root, dir_name)
                
                # 如果是中文文件夹，继续处理其中的子文件夹
                if is_chinese(dir_name):  # 判断是否为中文文件夹（可以根据需求更改此函数）
                    rename_subfolders_in_chinese_folder(chinese_folder_path, dir_name, pinyin_dict)

def path_exists_case_sensitive(path):
    """
    检查路径是否存在，并且区分大小写
    """
    directory, filename = os.path.split(path)
    
    # 遍历目录中的所有文件和文件夹，手动比较大小写
    try:
        # 获取目录下所有文件和文件夹
        for entry in os.listdir(directory):
            if entry == filename:
                return True
    except FileNotFoundError:
        return False
    
    return False

def rename_subfolders_in_chinese_folder(chinese_folder_path, chinese_folder_name, pinyin_dict):
    """
    将中文文件夹下的子文件夹重命名为中文文件夹的拼音，若包含 '_trans' 则保留后缀。
    
    参数:
    - chinese_folder_path: 中文文件夹路径
    - chinese_folder_name: 中文文件夹名称
    - pinyin_dict: 拼音字典，中文到拼音
    """
    # 获取中文文件夹下的所有子文件夹
    for sub_dir_name in os.listdir(chinese_folder_path):
        subfolder_path = os.path.join(chinese_folder_path, sub_dir_name)
        
        # 只处理文件夹
        if os.path.isdir(subfolder_path):
            # 判断子文件夹名是否包含 '_trans'
            if "_trans" in sub_dir_name:
                new_name = get_pinyin_from_dict(chinese_folder_name, pinyin_dict) + "_trans"  # 保留 _trans 后缀
            else:
                new_name = get_pinyin_from_dict(chinese_folder_name, pinyin_dict)  # 直接转换为拼音
            
            new_subfolder_path = os.path.join(chinese_folder_path, new_name)
            
            # 如果新名字与旧名字不同，且目标路径不存在，则进行重命名
            if subfolder_path != new_subfolder_path and not path_exists_case_sensitive(new_subfolder_path):  
                os.rename(subfolder_path, new_subfolder_path)
                print(f"Renamed {subfolder_path} to {new_subfolder_path}")
            else:
                print(f"Skipped renaming {subfolder_path}, target already exists or no change needed.")

def get_pinyin_from_dict(chinese_name, pinyin_dict):
    """
    根据中文名称从拼音字典中查找拼音。
    
    参数:
    - chinese_name: 中文名称
    - pinyin_dict: 拼音字典，中文到拼音
    
    返回:
    - 拼音字符串
    """
    return pinyin_dict.get(chinese_name, chinese_name)  # 如果字典中没有找到，返回原中文名称

def is_chinese(name):
    """
    判断文件夹名称是否包含中文字符。
    
    参数:
    - name: 文件夹名称
    返回:
    - 是否包含中文字符
    """
    return any('\u4e00' <= char <= '\u9fff' for char in name)

if __name__ == "__main__":
    # 指定YAML文件路径
    yaml_file = "config/characters.yaml"  # 修改为实际的YAML文件路径
    is_pinyin = False #设置为True会将中文拼音改为英文 Falas会将英文改为中文拼音
    pinyin_dict = load_dict(yaml_file, is_pinyin)

    # 指定要处理的根文件夹路径
    root_folder = r"E:\图包006_r"  # 修改为你的实际路径
    pinyin_dict, english_dict = load_dict_from_yaml(yaml_file)

    is_pinyin_to_english = False
    
    #rename_subfolders_with_pinyin(root_folder, pinyin_dict) #依赖上级中文目录在字典中匹配进行拼音重命名或者英文重命名 优点是可把命名错误的英文或者拼音使用字典依照上级目录进行纠正
    rename_folders_in_directory(root_folder, is_pinyin_to_english, pinyin_dict, english_dict) #依赖本身的文件名字在字典中匹配对应的英文名或拼音 优点是不依赖中文上级目录