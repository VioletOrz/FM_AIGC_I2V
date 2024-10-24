import argparse
import yaml
from datetime import datetime
import os
import time

def Args():
    parser = argparse.ArgumentParser(description="A script to process prompts and config files")

    # 添加参数
    parser.add_argument('--config', type=str, required=False, help='Path to the config file')

    # 解析命令行参数
    args = parser.parse_args()

    # 使用解析到的参数

    config_file = args.config

    if config_file == None:
        config_file = './config/sitting.yaml'

    return config_file

def argument(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
        original_image_path = config['Path']['original_image_path'] #原始图片的输入路径
        emotion_video = config['Path']['emotion_video'] #参考表情视频的文件名（包含扩展名）  
        
        background_path = config['Path']['background_path'] #用于合成的背景路径
        alternate_background_path = config['Path']['alternate_background_path']

        input_image_path = config['Path']['input_image_path'] #输入图像的路径(分割好之后不带背景的人物图片)
        mask_output_dir = config['Path']['mask_output_dir'] #没有背景的人物图像和遮罩的输出路径
        emotion_video_path = config['Path']['emotion_video_path'] #+a1#表情视频的路径
        background_output_video_path = config['Path']['background_output_video_path'] #加上背景的视频输出路径
        vsr_output_video_path = config['Path']['vsr_output_video_path'] #超分输出视频的路径

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
        #force_extension = config['mode']['force_extension']

        face_landmarker_path = config['model_path']['face_landmarker_path']
        

        return {        
            'Path': {
                'original_image_path': original_image_path,
                'emotion_video': emotion_video,
                'background_path': background_path,
                'alternate_background_path': alternate_background_path,
                'input_image_path': input_image_path,
                'mask_output_dir': mask_output_dir,
                'emotion_video_path': emotion_video_path,
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
    #######################

    original_image_path = config['Path']['original_image_path'] #原始图片的输入路径
    emotion_video = config['Path']['emotion_video'] #参考表情视频的文件名（包含扩展名）  
        
    background_path = config['Path']['background_path'] #用于合成的背景路径
    alternate_background_path = config['Path']['alternate_background_path']

    input_image_path = config['Path']['input_image_path'] #输入图像的路径(分割好之后不带背景的人物图片)
    mask_output_dir = config['Path']['mask_output_dir'] #没有背景的人物图像和遮罩的输出路径
    emotion_video_path = config['Path']['emotion_video_path'] + emotion_video  #+a1#表情视频的路径
    background_output_video_path = config['Path']['background_output_video_path'] #加上背景的视频输出路径
    cropped_dir = config['Path']['cropped_dir']

    face_landmarker_path = config['model_path']['face_landmarker_path']

    units_ins=units()
    #step1:抠出人物图片和遮罩(source_imgs->mask_output)
    units_ins.step1(original_image_path,mask_output_dir,cropped_dir)
    print("----------------------------step1 finished!!-------------------------------------")
    # step2：将遮罩图片和原始图片输入sd，重绘得到背景图片(source_imgs+mask_output->background)
    # units_ins.step2(cropped_path,mask_output_dir+'mask_0.png',background_path)
    # print("----------------------------step2 finished!!-------------------------------------")
    
    units_ins.step2(original_image_path,mask_output_dir+'mask_0.png',background_path)
    print("----------------------------step2 finished!!-------------------------------------")

    #step3:输入人物图片+表情视频，输出透明背景的人物表情图片(emotions+mask_output->emotion_output)
    #注意：step4已经被整合进了step3，为了提高合成速度，去掉了中间保存每一帧的图片的步骤。要注意，step3不能和step1分开运行，因为在拼接视频的时候，用到了step1的输出结果
    if alternate_background:
        from modules.Resize_image import resize_image_ref
        resize_image_ref(alternate_background_path,background_path,background_path)

    units_ins.step3(input_image_path,emotion_video_path,background_path,background_output_video_path,face_landmarker_path,is_trans)
    print("----------------------------step3 finished!!-------------------------------------")    

def Img_pack_6T(config, output_flie):

    import time
    import shutil
    from modules.Img_pack_6T import units2

    #######################
    is_trans = config['mode']['is_trans']
    alternate_background = config['mode']['alternate_background']
    #######################

    original_image_path = config['Path']['original_image_path'] #原始图片的输入路径

    #下面这些不用改
    input_image_path = config['Path']['input_image_path'] #输入图像的路径(分割好之后不带背景的人物图片)
    mask_output_dir = config['Path']['mask_output_dir'] #没有背景的人物图像和遮罩的输出路径

    emotion_video = config['Path']['emotion_video'] #参考表情视频的文件名（包含扩展名）
    background_path = config['Path']['background_path'] #用于合成的背景路径
    alternate_background_path = config['Path']['alternate_background_path']

    emotion_video_path = config['Path']['emotion_video_path'] + emotion_video  #+a1#表情视频的路径
    cropped_dir = config['Path']['cropped_dir']



    face_landmarker_path = config['model_path']['face_landmarker_path']

    units_ins2=units2()
 
    #step1:抠出人物图片和遮罩(source_imgs->mask_output)
    units_ins2.step1(original_image_path,mask_output_dir,cropped_dir)
    print("----------------------------step1 finished!!-------------------------------------")

    # units_ins2.step2(cropped_path,mask_output_dir+'mask_0.png',background_path)
    # print("----------------------------step2 finished!!-------------------------------------")
    units_ins2.step2(original_image_path,mask_output_dir+'mask_0.png',background_path)
    print("----------------------------step2 finished!!-------------------------------------")

    #step3:输入人物图片+表情视频，输出透明背景的人物表情图片(emotions+mask_output->emotion_output)
    #注意：step4已经被整合进了step3，为了提高合成速度，去掉了中间保存每一帧的图片的步骤。要注意，step3不能和step1分开运行，因为在拼接视频的时候，用到了step1的输出结果
    if alternate_background:
        from modules.Resize_image import resize_image_ref
        resize_image_ref(alternate_background_path,background_path,background_path)

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


def main():
    config_file = Args()
    config = argument(config_file)
    Preview(config)
    #Generate_image_package_pipeline_per_step(config)
    Generate_image_package_pipeline(config)

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

    
    output_base = Img_pack_6T(config, base_image_path)
    output_dehaze = Dehazing(config, output_base, dehazed_image_path)
    output_HD = SupResHD(config, output_dehaze, SuperResHD_image_path)
    output_resize = Resize_batch(config, output_HD, resized_image_path)
    Difference_image(config, output_HD, DiffHD_image_path)
    Difference_image(config, output_resize, DiffLD_image_path)

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

    Img_pack_6T(config, base_image_path)
    #SupResHD(config, base_image_path, SuperResHD_image_path)
    Dehazing(config, base_image_path, dehazed_image_path)
    #SupResHD(config, dehazed_image_path, SuperResHD_image_path)
    #Resize_batch(config, SuperResHD_image_path, resized_image_path)
    Difference_image(config, dehazed_image_path, DiffHD_image_path)
    #Difference_image(config, SuperResHD_image_path, DiffHD_image_path)
    #Difference_image(config, resized_image_path, DiffLD_image_path)



if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {int(elapsed_time//60)} 分 {elapsed_time%60} 秒")