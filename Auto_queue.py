import subprocess
import yaml
import os
import time
from tqdm import tqdm
import shutil
from tools.move_to_ch_dir import move_matching_folders, copy_files_to_directories
from launch import Args,argument, Generate_image_package_pipeline_from_String
import torch

def load_characters_yaml(characters_path, is_pinyin = False):
    with open(characters_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
        if is_pinyin: character_name = config["character_name_pinyin"]
        else: character_name = config["character_name_English"]
        characters = {
            "star_indx": config["star_indx"],
            "input_paths": config["image_path"],
            "package_name": character_name,
        }
    return characters

def load_config_yaml(config_path):
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)

        mask_output_dir = os.path.basename(os.path.normpath(config['Path']['mask_output_dir'] ))
        background_output_video_path = os.path.basename(os.path.normpath(config['Path']['background_output_video_path'])) 
        cropped_dir = os.path.basename(os.path.normpath(config['Path']['cropped_dir']))
            
        img_pack_save_path = config['Path']['img_pack_save_path']
    return mask_output_dir, background_output_video_path, cropped_dir, img_pack_save_path

# 循环执行每个命令
def Auto_queue_pipeline(characters, config_path, pipeline, is_trans_list, alternate_background, emotion_from_tensor,):

    mask_output_dir, background_output_video_path, cropped_dir, img_pack_save_path = load_config_yaml(config_path)

    all_cnt =  len(characters["input_paths"][characters["star_indx"]:]) * len(is_trans_list)
    cnt = 0
    c_cnt = len(characters["input_paths"][characters["star_indx"]:])
    print("###############################################")
    print(f"角色数为：{c_cnt}")
    print(f"总图包数为: {all_cnt}")
    print("###############################################")

    total = all_cnt  # 总任务数量
    outer_bar = tqdm(range(total), desc="正在生成图包", unit="task")

    #arg = Args()

    arg = {
        "config": config_path,
        "is_trans": None,
        'package_name': None,
        'input_path': None,
        'alternate_background': None,
        'alternate_background_path': None,
        'emotion_from_tensor': None,
        'emotion_pose_save_path': None,
        'emotion_pose_load_path': None,
    }
    
    config = argument(arg)

    for input_path, package_name in zip(characters["input_paths"][characters["star_indx"]:],characters["package_name"][characters["star_indx"]:]) :
        for is_trans in is_trans_list:
            cnt = cnt + 1
            #test
            #if cnt == 1 : return
            #params = {
            #    "python": "./launch.py",
            #    "--config": config_path,
            #    "--input_path": input_path,
            #    "--pipeline": pipeline,
            #    "--is_trans": is_trans,
            #    "--package_name": package_name,
            #    "--alternate_backgroud": alternate_background,
            #    "--emotion_from_tensor": emotion_from_tensor
            #}
            print()
            print("###############################################")
            print(f"当前生成角色为: {package_name}, is_trans = {is_trans} ")
            start_time = time.time()

            config['Path']['img_pack_file_name'] = package_name
            config['Path']['original_image_path'] = input_path

            if is_trans == 'True': is_trans = True
            elif is_trans == 'False': is_trans = False
            config['mode']['is_trans'] = is_trans

            if alternate_background == 'True': alternate_background = True
            elif alternate_background == 'False': alternate_background = False
            config['mode']['alternate_background'] = alternate_background
            config['mode']['emotion_from_tensor'] = emotion_from_tensor
            
            Generate_image_package_pipeline_from_String(config, pipeline)
            torch.cuda.empty_cache()

            #print("\033c", end="")
            
            #cmd = ["python", "launch.py"] + [str(item) for pair in params.items() for item in pair]
            # 使用 subprocess.run 来执行命令
            #result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            # 打印输出和错误信息 
            #print(f"Running command with input_path: {input_path}")
            #print("Output:", result.stdout)
            #print("Error:", result.stderr)
            #print("Exit Status:", result.returncode)
            print("-" * 50)

            #在24.11.15的更新中这段代码被加入到了pipeline中 请参考Log相关说明使用 在pipeline中加入C/c以进行代替
            #from tools.copy_select_img_pack import copy_specified_folders_with_structure
            #src_folder = "./data/"
            #if is_trans == "True":
            #    target_folder = img_pack_save_path + f"{package_name}_trans"
            #else:
            #    target_folder = img_pack_save_path + f"{package_name}"
            #folder_names = [str(mask_output_dir),str(background_output_video_path),str(cropped_dir)]  # 可以指定多个文件夹名称
            #copy_specified_folders_with_structure(src_folder, target_folder, folder_names)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("###############################################")
            print(f"Indx:{cnt - 1} 第{cnt}个 图包{package_name}, is_trans = {is_trans}用时: {int(elapsed_time//60)} 分 {elapsed_time%60} 秒")
            print(f"总进度：{cnt}/{all_cnt}")
            print("###############################################")
            outer_bar.update(1)

def main():
    #参数设置
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    characters_path = "./config/characters.yaml"
    config_path = "./config/sitting.yaml"

    pipeline = "PIFDSRFC"
    is_trans_list = ["True", "False"]
    alternate_background = "False"
    emotion_from_tensor = "True"

    target_root = "E:/图包/006"

    is_pinyin = False #是否使用拼音代替英文命名 建议设置为false如果需要中文命名可以使用 tools\Eng_to_Ch_pinyin.py

    characters = load_characters_yaml(characters_path, is_pinyin)
    if len(characters["input_paths"]) == len(characters["package_name"]):
        image_list_len = len(characters["input_paths"][characters["star_indx"]:])
        print("###############################################")
        print(f"总图片数为: {image_list_len}")
        print("###############################################")
    else:
        print("###############################################")
        print("图片数与包体名称对应有误，请检查配置文件")
        print("###############################################")

    _, _, _, img_pack_save_path = load_config_yaml(config_path)

    Auto_queue_pipeline(characters, config_path, pipeline, is_trans_list, alternate_background, emotion_from_tensor, )

    print("###############################################")
    print("队列已完成")
    print("###############################################")
    print("正在将图包移动至中文目录...")
    #移动图包至中文目录
    move_matching_folders(characters_path, img_pack_save_path, target_root, is_pinyin)
    #复制源图像到中文目录
    copy_files_to_directories(characters_path, target_root, is_pinyin)
    print("###############################################")
        
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("###############################################")
    print(f"代码运行时间: {int(elapsed_time//60)} 分 {elapsed_time%60} 秒")
    print("###############################################")