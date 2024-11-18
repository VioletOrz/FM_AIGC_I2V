import argparse
import yaml
from datetime import datetime
import os
import time
from tools.copy_select_img_pack import copy_specified_folders_with_structure
import torch

def Args():
    parser = argparse.ArgumentParser(description="A script to process config")

    # 添加参数
    parser.add_argument('--config', default="./config/sitting.yaml", type=str, required=False, help='Path to the config file.')
    parser.add_argument('--pipeline', type=str, default="PIFDFSFRF",required=False, help='Pipeline.')
    parser.add_argument('--is_trans', type=str, default=None, required=False, help='Generate transparent no background image package.')
    parser.add_argument('--package_name', type=str, default=None, required=False, help='Output package name.')
    parser.add_argument('--input_path', type=str, default=None, required=False, help='Input image path.')
    parser.add_argument('--alternate_background', type=str, default=None, required=False, help='Generate image package with alternate background.')
    parser.add_argument('--alternate_background_path', type=str, default=None, required=False, help='Alternate background path.')
    parser.add_argument('--emotion_from_tensor', type=str, default=None, required=False, help='Read the processed pose data set from a.pt file.')
    parser.add_argument('--emotion_pose_save_path', type=str, default=None, required=False, help='Processed pose data save path.')
    parser.add_argument('--emotion_pose_load_path', type=str, default=None, required=False, help='Processed pose data load path.')
    
    # 解析命令行参数
    args, unknown_args = parser.parse_known_args()

    # 使用解析到的参数

    config_file = args.config
    pipeline_string = args.pipeline
    mode_is_trans = args.is_trans
    package_name = args.package_name
    input_path = args.input_path
    alternate_background = args.alternate_background
    alternate_background_path = args.alternate_background_path
    emotion_from_tensor = args.emotion_from_tensor
    emotion_pose_save_path = args.emotion_pose_save_path
    emotion_pose_load_path = args.emotion_pose_load_path


    print({
        "config": config_file,
        "pipeline": pipeline_string,
        "is_trans": mode_is_trans,
        'package_name': package_name,
        'input_path': input_path,
        'alternate_background': alternate_background,
        'alternate_background_path': alternate_background_path,
        'emotion_from_tensor':emotion_from_tensor,
        'emotion_pose_save_path': emotion_pose_save_path,
        'emotion_pose_load_path': emotion_pose_load_path,
    })

    return {
        "config": config_file,
        "pipeline": pipeline_string,
        "is_trans": mode_is_trans,
        'package_name': package_name,
        'input_path': input_path,
        'alternate_background': alternate_background,
        'alternate_background_path': alternate_background_path,
        'emotion_from_tensor': emotion_from_tensor,
        'emotion_pose_save_path': emotion_pose_save_path,
        'emotion_pose_load_path': emotion_pose_load_path,
    }

def argument(arg = {}):
    config_path = arg['config']
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
        original_image_path = config['Path']['original_image_path']

        emotion_video = config['Path']['emotion_video'] 
        emotion_video_path = config['Path']['emotion_video_path'] 
        emotion_pose_save_path = config['Path']['emotion_pose_save_path']
        emotion_pose_load_path = config['Path']['emotion_pose_load_path']

        background_path = config['Path']['background_path'] 
        alternate_background_path = config['Path']['alternate_background_path']

        input_image_path = config['Path']['input_image_path'] 
        mask_output_dir = config['Path']['mask_output_dir'] 
        
        background_output_video_path = config['Path']['background_output_video_path'] 
        vsr_output_video_path = config['Path']['vsr_output_video_path'] 

        cropped_dir = config['Path']['cropped_dir']
        #cropped_path = config['Path']['cropped_path']
        
        img_pack_save_path = config['Path']['img_pack_save_path']
        img_pack_file_name = config['Path']['img_pack_file_name']
        base_pack_name = config['Path']['base_pack_name']
        dehazed_pack_name = config['Path']['dehazed_pack_name']
        SuperResHD_pack_name = config['Path']['SuperResHD_pack_name']
        resized_pack_name = config['Path']['resized_pack_name']
        DiffHD_pack_name = config['Path']['DiffHD_pack_name']
        DiffLD_pack_name = config['Path']['DiffLD_pack_name']

        is_trans = config['mode']['is_trans']
        alternate_background = config['mode']['alternate_background']
        emotion_from_tensor = config['mode']['emotion_from_tensor']
        #force_extension = config['mode']['force_extension']

        face_landmarker_path = config['model_path']['face_landmarker_path']
    
    if arg['is_trans'] != None:
        if arg['is_trans'] == "True":
            is_trans = True
        elif arg['is_trans'] == "False":
            is_trans = False
    if arg['package_name'] != None:
        img_pack_file_name = arg['package_name']
    if arg['input_path'] != None:
        original_image_path = arg['input_path']
    if arg['alternate_background'] != None:
        if arg['alternate_background'] == "True":
            alternate_background = True
        elif arg['alternate_background'] == "False":
            alternate_background = False
    if arg['alternate_background_path'] != None:
        alternate_background_path = arg['alternate_background_path']
    if arg['emotion_from_tensor'] != None:
        if arg['emotion_from_tensor'] == "True":
            emotion_from_tensor = True
        elif arg['emotion_from_tensor'] == "False":
            emotion_from_tensor = False
    if arg['emotion_pose_save_path'] != None:
        emotion_pose_save_path = arg['emotion_pose_save_path']
    if arg['emotion_pose_load_path'] != None:
        emotion_pose_load_path = arg['emotion_pose_load_path']  
    

    return {        
        'Path': {
                'original_image_path': original_image_path,
                'emotion_video': emotion_video,
                'background_path': background_path,
                'alternate_background_path': alternate_background_path,
                'input_image_path': input_image_path,
                'mask_output_dir': mask_output_dir,
                'emotion_video_path': emotion_video_path,
                'emotion_pose_save_path': emotion_pose_save_path,
                'emotion_pose_load_path': emotion_pose_load_path,
                'background_output_video_path': background_output_video_path,
                'vsr_output_video_path': vsr_output_video_path,
                'cropped_dir': cropped_dir,
                #'cropped_path': cropped_path,
                'img_pack_save_path': img_pack_save_path,
                'base_pack_name': base_pack_name,
                'dehazed_pack_name': dehazed_pack_name,
                'SuperResHD_pack_name': SuperResHD_pack_name,
                'resized_pack_name': resized_pack_name,
                'DiffHD_pack_name': DiffHD_pack_name,
                'DiffLD_pack_name': DiffLD_pack_name,
                'img_pack_file_name': img_pack_file_name,
            },
        'mode':{
                'is_trans': is_trans,
                'alternate_background': alternate_background,
                'emotion_from_tensor': emotion_from_tensor,
            },
        'model_path': {
                'face_landmarker_path': face_landmarker_path,
                #'force_extension': force_extension,
            }
    }

def save_image(image, save_path: str):

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace('-', '').replace(':', '').replace(' ','')
    image.save(f"{save_path}{current_time}.png")

def Preview(config):
    from modules.Preview import units

    #######################
    is_trans = config['mode']['is_trans']
    alternate_background = config['mode']['alternate_background']
    emotion_from_tensor = config['mode']['emotion_from_tensor']
    #######################

    original_image_path = config['Path']['original_image_path']
    emotion_video = config['Path']['emotion_video'] 
    emotion_pose_load_path = config['Path']['emotion_pose_load_path']

    background_path = config['Path']['background_path'] 
    alternate_background_path = config['Path']['alternate_background_path']

    input_image_path = config['Path']['input_image_path'] 
    mask_output_dir = config['Path']['mask_output_dir'] 
    emotion_video_path = config['Path']['emotion_video_path'] + emotion_video 
    background_output_video_path = config['Path']['background_output_video_path']
    cropped_dir = config['Path']['cropped_dir']

    face_landmarker_path = config['model_path']['face_landmarker_path']

    units_ins=units()
    
    units_ins.step1(original_image_path,mask_output_dir,cropped_dir)
    print("----------------------------step1 finished!!-------------------------------------")

    if is_trans == False:
        units_ins.step2(original_image_path,mask_output_dir+'mask_0.png',background_path,is_trans,alternate_background)
    print("----------------------------step2 finished!!-------------------------------------")

    if is_trans == False and alternate_background:
        from modules.Resize_image import resize_image_ref
        resize_image_ref(alternate_background_path,original_image_path,background_path)

    if is_trans:
        background_path = original_image_path

    if emotion_from_tensor:
        units_ins.step3_pose_from_tensor(input_image_path,emotion_pose_load_path,background_path,background_output_video_path,face_landmarker_path, is_trans)
    else:
        units_ins.step3(input_image_path,emotion_video_path,background_path,background_output_video_path,face_landmarker_path,is_trans)
    print("----------------------------step3 finished!!-------------------------------------")    

def Gen_img_pack(config, output_flie):

    import time
    import shutil
    from modules.Gen_img_pack import units2

    #######################
    is_trans = config['mode']['is_trans']
    alternate_background = config['mode']['alternate_background']
    emotion_from_tensor = config['mode']['emotion_from_tensor']
    #######################

    original_image_path = config['Path']['original_image_path']

    input_image_path = config['Path']['input_image_path'] 
    mask_output_dir = config['Path']['mask_output_dir'] 

    emotion_video = config['Path']['emotion_video']
    emotion_pose_load_path = config['Path']['emotion_pose_load_path']
     
    background_path = config['Path']['background_path'] 
    alternate_background_path = config['Path']['alternate_background_path']

    emotion_video_path = config['Path']['emotion_video_path'] + emotion_video  
    cropped_dir = config['Path']['cropped_dir']



    face_landmarker_path = config['model_path']['face_landmarker_path']

    units_ins2=units2()

    units_ins2.step1(original_image_path,mask_output_dir,cropped_dir)
    print("----------------------------step1 finished!!-------------------------------------")

    if is_trans == False:
        units_ins2.step2(original_image_path,mask_output_dir+'mask_0.png',background_path,is_trans,alternate_background)
    print("----------------------------step2 finished!!-------------------------------------")

    if is_trans == False and alternate_background:
        from modules.Resize_image import resize_image_ref
        resize_image_ref(alternate_background_path,original_image_path,background_path)

    if is_trans:
        background_path = original_image_path

    if emotion_from_tensor:
        units_ins2.step3_pose_from_tensor(input_image_path,emotion_pose_load_path,background_path,output_flie,face_landmarker_path, is_trans)
    else: 
        units_ins2.step3(input_image_path,emotion_video_path,background_path,output_flie,face_landmarker_path,is_trans)
    print("----------------------------step3 finished!!-------------------------------------")

    path1=output_flie+"/o_0.5"
    shutil.rmtree(path1)
    path2=output_flie+"/i_0.5"
    shutil.rmtree(path2)

    return output_flie

def Dehazing(config,input_file, output_file):

    from modules.Dehaze import dehaze_V2

    transform_mode=['b','a','i','o']
    transform_strength=[1.0,0.5]

    for mode in transform_mode:
        print(f"###################################################开始去朦胧{mode}表情###################################################")
        if mode=='b':
            strength=0.0
            print(f"###################################################开始去朦胧{mode}表情的强度为{strength}###################################################")
            if not os.path.exists(output_file+f'/{mode}_{strength}'):
                os.makedirs(output_file+f'/{mode}_{strength}')
            image_path = input_file+f'/{mode}_{strength}'
            print(image_path)
            output_path = output_file+f'/{mode}_{strength}'
            dehaze_V2(image_path,output_path)
        else:
            for strength in transform_strength:
                if  mode=='i' or mode=='o':
                    strength=1.0
                print(f"###################################################开始去朦胧{mode}表情的强度为{strength}###################################################")
                if not os.path.exists(output_file+f'/{mode}_{strength}'):
                    os.makedirs(output_file+f'/{mode}_{strength}')
                image_path = input_file+f'/{mode}_{strength}'
                print(image_path)
                output_path = output_file+f'/{mode}_{strength}'
                dehaze_V2(image_path,output_path)
    
    return output_file

def SupResHD(config,input_file, output_file):
    from modules.SuperResHD import super_resolution
    super_resolution(input_file, output_file)
    return output_file

def Resize_batch(config,input_file, output_file):
    from modules.Resize_image import resize_batch
    resize_batch(input_file, output_file)
    return output_file

def Difference_image(config,input_file, output_file):
    from modules.Difference_image import diff_image
    diff_image(input_file, output_file)
    return output_file

def process_emotion_video_to_pose_tensor(config):
    from modules.Preview import units
    from modules.Gen_img_pack import units2

    original_image_path = config['Path']['original_image_path']
    emotion_video = config['Path']['emotion_video'] 
    emotion_video_path = config['Path']['emotion_video_path'] + emotion_video
    background_output_video_path = config['Path']['background_output_video_path']
    emotion_pose_save_path = config['Path']['emotion_pose_save_path']
    face_landmarker_path = config['model_path']['face_landmarker_path']

    units_preview=units()
    units_preview.process_and_save_emotion_pose(emotion_video_path, background_output_video_path, face_landmarker_path, emotion_pose_save_path)
    units_ins=units2()
    units_ins.process_and_save_emotion_pose(emotion_video_path, face_landmarker_path, emotion_pose_save_path)

def main():
    arg = Args()
    config = argument(arg)
    #process_emotion_video_to_pose_tensor(config)
    Generate_image_package_pipeline_from_String(config, arg['pipeline'])
    #Preview(config)
    #Generate_image_package_pipeline_per_step(config)
    #Generate_image_package_pipeline(config)

def Generate_image_package_pipeline_from_String(config, pipeline: str):
    #PIDSRF
    #P:Preview
    #I:Image package generate
    #D:Dehaze
    #S:SuperResHD
    #R:Resize
    #F:Final-->Generate Difference Image Package
    img_pack_save_path = config['Path']['img_pack_save_path']
    img_pack_file_name = config['Path']['img_pack_file_name']

    #######################
    is_trans = config['mode']['is_trans']
    #######################
    if is_trans: img_pack_file_name = img_pack_file_name + '_trans'

    base_pack_name = config['Path']['base_pack_name']
    base_image_path = img_pack_save_path + img_pack_file_name + '/' + base_pack_name
    dehazed_pack_name = config['Path']['dehazed_pack_name']
    dehazed_image_path = img_pack_save_path + img_pack_file_name + '/' + dehazed_pack_name
    SuperResHD_pack_name = config['Path']['SuperResHD_pack_name']
    SuperResHD_image_path = img_pack_save_path + img_pack_file_name + '/' + SuperResHD_pack_name
    resized_pack_name = config['Path']['resized_pack_name']
    resized_image_path = img_pack_save_path + img_pack_file_name + '/' + resized_pack_name

    sys_output = ''
    cnt  = 0
    for process_step in pipeline:
        if process_step not in "PIDSRFCpidsrc":
            print("###################################################################################################")
            print(f"The pipeline char is must in P, I/i, D/d, S/s, R/r, F, C/c, the input pipeline char is: {process_step}.")
            print("###################################################################################################")
            return
        if process_step == 'p':
            if cnt != 0 :
                print("###################################################################################################")
                print(f"The pipeline is noncompliant, p must on top.")
                print("###################################################################################################")
                return
            sys_output = sys_output + 'Process emotion video to pose tensor --> '
        if process_step == 'P':
            if cnt != 0 and pipeline[cnt - 1] != 'p':
                print("###################################################################################################")
                print(f"The pipeline is noncompliant, P must on top or after p.")
                print("###################################################################################################")
                return
            sys_output = sys_output + 'Preview --> '
        elif process_step == 'I':
            if cnt != 0 :
                if cnt > 2:
                    print("###################################################################################################")
                    print(f"The pipeline is noncompliant, I must on top or afer P/p")
                    print("###################################################################################################")
                    return
                if cnt == 1 and pipeline[cnt - 1] != 'P' and pipeline[cnt - 1] != 'p':
                    print("###################################################################################################")
                    print(f"The pipeline is noncompliant, I must on top or afer P/p")
                    print("###################################################################################################")
                    return
                if cnt == 2 and pipeline[cnt - 1] != 'P':
                    print("###################################################################################################")
                    print(f"The pipeline is noncompliant, I must on top or afer P/p")
                    print("###################################################################################################")
                    return
            sys_output = sys_output + 'Generate Image package --> '
        elif process_step == 'D':
            if cnt != 0 :
                last = pipeline[cnt - 1]
                if last not in "IiF":
                    print("###################################################################################################")
                    print(f"The pipeline is noncompliant, D must after I/i or F.")
                    print("###################################################################################################")
                    return
            if cnt - 1 < 0:
                print("###################################################################################################")
                print(f"The pipeline is noncompliant, D must after I/i or F.")
                print("###################################################################################################")
                return
            sys_output = sys_output + 'Dehazing process --> '
        elif process_step == 'S':
            if cnt != 0 :
                last = pipeline[cnt - 1]
                if last not in "IiDdF":
                    print("###################################################################################################")
                    print(f"The pipeline is noncompliant, S must after I/i, D/d, or F.")
                    print("###################################################################################################")
                    return
            if cnt - 1 < 0:
                print("###################################################################################################")
                print(f"The pipeline is noncompliant, S must after I/i, D/d, or F.")
                print("###################################################################################################")
                return
            sys_output = sys_output + 'SuperRes HD process --> '
        elif process_step == 'R':
            if cnt != 0 :
                last = pipeline[cnt - 1]
                if last not in "SsF":
                    print("###################################################################################################")
                    print(f"The pipeline is noncompliant, R must after S/s or F.")
                    print("###################################################################################################")
                    return
            if cnt - 1 < 0:
                print("###################################################################################################")
                print(f"The pipeline is noncompliant, R must after S/s or F.")
                print("###################################################################################################")
                return
            sys_output = sys_output + 'Resize H and W to half Res --> '
        elif process_step == 'F':
            if cnt != 0 :
                last = pipeline[cnt - 1]
                if last not in "IDSRidsr":
                    print("###################################################################################################")
                    print(f"The pipeline is noncompliant, F must after I/i, D/d, S/s, or R/r.")
                    print("###################################################################################################")
                    return
            if cnt - 1 < 0:
                print("###################################################################################################")
                print(f"The pipeline is noncompliant, F must after I/i, D/d, S/s, or R/r.")
                print("###################################################################################################")
                return
            sys_output = sys_output + f'Generate {last} Difference Image package.  ||  '
        elif process_step in "idsr":
            if cnt != 0 :
                print("###################################################################################################")
                print(f"The pipeline is noncompliant, {process_step} must on top.")
                print("###################################################################################################")
                return
            sys_output = sys_output + f'From last Interrupt Step {process_step} Continue --> '
        elif process_step == 'C':
            if cnt != len(pipeline) - 1:
                print("###################################################################################################")
                print(f"The pipeline is noncompliant, C must on bottom .")
                print("###################################################################################################")
                return
            if 'P' not in pipeline:
                print("###################################################################################################")
                print(f"The pipeline is noncompliant, When c is used, P must be in the pipeline .")
                print(f"If you wish to skip the 'P' Pipeline check, use c instead.")
                print(f"When using, make sure that the P process that was last run is the one with the current input image.")
                print("###################################################################################################")
                return
            sys_output = sys_output + 'Copy "cropdir" "background_output" and "mask_output" to output path --> '
        elif process_step == 'c':
            if cnt != len(pipeline) - 1:
                print("###################################################################################################")
                print(f"The pipeline is noncompliant, c must on bottom .")
                print("###################################################################################################")
                return
            sys_output = sys_output + 'Copy #the latest#!!! "cropdir" "background_output" and "mask_output" to output path --> '
        cnt = cnt + 1

    print("###########################################")
    print(sys_output)
    print("###########################################")
    time.sleep(1)


    last_output_path = None
    cnt = 0
    for process_step in pipeline:
        if process_step == 'p':
            process_emotion_video_to_pose_tensor(config)
        elif process_step == 'P':
            Preview(config)
        elif process_step == 'I':
            last_output_path = Gen_img_pack(config = config, output_flie = base_image_path)
            torch.cuda.empty_cache()
        elif process_step == 'D':
            if last_output_path == None:
                last_output_path = continue_path(config, pipeline[cnt-1])
            last_output_path = Dehazing(config = config, input_file = last_output_path, output_file = dehazed_image_path)
        elif process_step == 'S':
            if last_output_path == None:
                last_output_path = continue_path(config, pipeline[cnt-1])
            last_output_path = SupResHD(config = config, input_file = last_output_path, output_file = SuperResHD_image_path)
            torch.cuda.empty_cache()
        elif process_step == 'R':
            if last_output_path == None:
                last_output_path = continue_path(config, pipeline[cnt-1])
            last_output_path = Resize_batch(config = config, input_file = last_output_path, output_file = resized_image_path)
        elif process_step == 'F':
            last_process_step = pipeline[cnt-1]
            if last_output_path == None:
                last_output_path = continue_path(config, pipeline[cnt-1])
                last_process_step = pipeline[cnt-1].upper()
            Difference_image(config = config, input_file = last_output_path, output_file = img_pack_save_path + img_pack_file_name + '/Difference_' + last_process_step)
        elif process_step == 'C' or process_step == 'c':
            mask_output_dir_name = os.path.basename(os.path.normpath(config['Path']['mask_output_dir'] ))
            background_output_video_path_name = os.path.basename(os.path.normpath(config['Path']['background_output_video_path'])) 
            cropped_dir_name = os.path.basename(os.path.normpath(config['Path']['cropped_dir']))
            folder_names = [str(mask_output_dir_name),str(background_output_video_path_name),str(cropped_dir_name)]  # 可以指定多个文件夹名称
            src_folder = "./data/"
            package_name = config['Path']['img_pack_file_name']
            if is_trans == True:
                target_folder = img_pack_save_path + f"{package_name}_trans"
            else:
                target_folder = img_pack_save_path + f"{package_name}"
            copy_specified_folders_with_structure(src_folder, target_folder, folder_names)
        cnt = cnt + 1
        #return

def continue_path(config, process_step):
    img_pack_save_path = config['Path']['img_pack_save_path']
    img_pack_file_name = config['Path']['img_pack_file_name']

    #######################
    is_trans = config['mode']['is_trans']
    #######################
    if is_trans: img_pack_file_name = img_pack_file_name + '_trans'

    base_pack_name = config['Path']['base_pack_name']
    base_image_path = img_pack_save_path + img_pack_file_name + '/' + base_pack_name
    dehazed_pack_name = config['Path']['dehazed_pack_name']
    dehazed_image_path = img_pack_save_path + img_pack_file_name + '/' + dehazed_pack_name
    SuperResHD_pack_name = config['Path']['SuperResHD_pack_name']
    SuperResHD_image_path = img_pack_save_path + img_pack_file_name + '/' + SuperResHD_pack_name
    resized_pack_name = config['Path']['resized_pack_name']
    resized_image_path = img_pack_save_path + img_pack_file_name + '/' + resized_pack_name

    if process_step == 'i':
        return base_image_path
    elif process_step == 'd':
        return dehazed_image_path
    elif process_step == 's':
        return SuperResHD_image_path
    elif process_step == 'r':
        return resized_image_path

def Generate_image_package_pipeline_per_step(config):


    #If the code breaks at a certain step, 
    # you can use this function to pick up where you left off, 
    # remember to adjust the input output and the functions you need to run!

    img_pack_save_path = config['Path']['img_pack_save_path']
    img_pack_file_name = config['Path']['img_pack_file_name']

    #######################
    is_trans = config['mode']['is_trans']
    #######################
    if is_trans: img_pack_file_name = img_pack_file_name + '_trans'

    base_pack_name = config['Path']['base_pack_name']
    base_image_path = img_pack_save_path + img_pack_file_name + '/' + base_pack_name
    dehazed_pack_name = config['Path']['dehazed_pack_name']
    dehazed_image_path = img_pack_save_path + img_pack_file_name + '/' + dehazed_pack_name
    SuperResHD_pack_name = config['Path']['SuperResHD_pack_name']
    SuperResHD_image_path = img_pack_save_path + img_pack_file_name + '/' + SuperResHD_pack_name
    resized_pack_name = config['Path']['resized_pack_name']
    resized_image_path = img_pack_save_path + img_pack_file_name + '/' + resized_pack_name
    DiffHD_pack_name = config['Path']['DiffHD_pack_name']
    DiffHD_image_path = img_pack_save_path + img_pack_file_name + '/' + DiffHD_pack_name
    DiffLD_pack_name = config['Path']['DiffLD_pack_name']
    DiffLD_image_path = img_pack_save_path + img_pack_file_name + '/' + DiffLD_pack_name

    #Img_pack_6T(config, base_image_path)
    #SupResHD(config, base_image_path, SuperResHD_image_path)
    #Dehazing(config, base_image_path, dehazed_image_path)
    #SupResHD(config, dehazed_image_path, SuperResHD_image_path)
    #Resize_batch(config, SuperResHD_image_path, resized_image_path)
    #Difference_image(config, base_image_path, DiffHD_image_path)
    #Difference_image(config, dehazed_image_path, DiffHD_image_path)
    #Difference_image(config, SuperResHD_image_path, DiffHD_image_path)
    #Difference_image(config, resized_image_path, DiffLD_image_path)

def Generate_image_package_pipeline(config):

    #Process all image output path
    img_pack_save_path = config['Path']['img_pack_save_path']
    img_pack_file_name = config['Path']['img_pack_file_name']

    #######################
    is_trans = config['mode']['is_trans']
    #######################
    if is_trans: img_pack_file_name = img_pack_file_name + '_trans'

    base_pack_name = config['Path']['base_pack_name']
    base_image_path = img_pack_save_path + img_pack_file_name + '/' + base_pack_name
    dehazed_pack_name = config['Path']['dehazed_pack_name']
    dehazed_image_path = img_pack_save_path + img_pack_file_name + '/' + dehazed_pack_name
    SuperResHD_pack_name = config['Path']['SuperResHD_pack_name']
    SuperResHD_image_path = img_pack_save_path + img_pack_file_name + '/' + SuperResHD_pack_name
    resized_pack_name = config['Path']['resized_pack_name']
    resized_image_path = img_pack_save_path + img_pack_file_name + '/' + resized_pack_name
    DiffHD_pack_name = config['Path']['DiffHD_pack_name']
    DiffHD_image_path = img_pack_save_path + img_pack_file_name + '/' + DiffHD_pack_name
    DiffLD_pack_name = config['Path']['DiffLD_pack_name']
    DiffLD_image_path = img_pack_save_path + img_pack_file_name + '/' + DiffLD_pack_name

    is_trans = config['mode']['is_trans']

    
    output_base = Gen_img_pack(config, base_image_path)
    output_dehaze = Dehazing(config, output_base, dehazed_image_path)
    output_HD = SupResHD(config, output_dehaze, SuperResHD_image_path)
    output_resize = Resize_batch(config, output_HD, resized_image_path)
    Difference_image(config, output_HD, DiffHD_image_path)
    Difference_image(config, output_resize, DiffLD_image_path)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {int(elapsed_time//60)} 分 {elapsed_time%60} 秒")