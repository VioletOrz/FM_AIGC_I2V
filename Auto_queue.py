import subprocess
import yaml
import os
import shutil
from tools.move_to_ch_dir import move_matching_folders, copy_files_to_directories

def load_characters_yaml(characters_path):
    with open(characters_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
        characters = {
            "input_paths": config["image_path"],
            "package_name": config["character_name"]
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

    all_cnt =  len(characters["input_paths"]) * len(is_trans_list)
    cnt = 0
    print("###############################################")
    print(f"总图包数为: {all_cnt}")
    print("###############################################")

    for input_path, package_name in zip(characters["input_paths"],characters["package_name"]) :
        for is_trans in is_trans_list:
            cnt = cnt + 1
            #test
            #if cnt == 1 : return
            params = {
                "python": "./launch.py",
                "--config": config_path,
                "--input_path": input_path,
                "--pipeline": pipeline,
                "--is_trans": is_trans,
                "--package_name": package_name,
                "--alternate_backgroud": alternate_background,
                "--emotion_from_tensor": emotion_from_tensor
            }

            print("###############################################")
            print(f"当前生成角色为: {package_name}, is_trans = {is_trans} ")
            
            
            cmd = ["python", "launch.py"] + [str(item) for pair in params.items() for item in pair]
            # 使用 subprocess.run 来执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            # 打印输出和错误信息 
            #print(f"Running command with input_path: {input_path}")
            #print("Output:", result.stdout)
            #print("Error:", result.stderr)
            #print("Exit Status:", result.returncode)
            print("-" * 50)

            from tools.copy_select_img_pack import copy_specified_folders_with_structure
            src_folder = "./data/"
            if is_trans == "True":
                target_folder = img_pack_save_path + f"{package_name}_trans"
            else:
                target_folder = img_pack_save_path + f"{package_name}"
            folder_names = [str(mask_output_dir),str(background_output_video_path),str(cropped_dir)]  # 可以指定多个文件夹名称
            copy_specified_folders_with_structure(src_folder, target_folder, folder_names)


            print(f"总进度：{cnt}/{all_cnt}")
            print("###############################################")

            
        
if __name__ == "__main__":
    #参数设置
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    characters_path = "./config/characters.yaml"
    config_path = "./config/sitting.yaml"

    pipeline = "PI"
    is_trans_list = ["True", "False"]
    alternate_background = "False"
    emotion_from_tensor = "True"

    target_root = "E:/图包/006"

    characters = load_characters_yaml(characters_path)
    if len(characters["input_paths"]) == len(characters["package_name"]):
        image_list_len = len(characters["input_paths"])
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

    move_matching_folders(characters_path, img_pack_save_path, target_root)
    copy_files_to_directories(characters_path, target_root)
    print("###############################################")