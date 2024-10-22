###########注意事项############
# 1. 本程序因为环境的问题，使用了很多绝对路径，所以在使用时需要根据自己的环境修改路径！！！其中stable diffusion本体的代码中也使用了绝对路径，所以要使用的话也需要修改路径
# 2. 如果不想修改路径，可以参考以下的路径进行配置：
#    将FaceMind_AIGC的内容直接放在D盘根目录下
#    将stable diffusion-master放在D:/下
# 3. 使用时将原始图像放在tha3/app/source_imgs下，名字改成img_0.jpg(你可以在下面自行修改)，表情视频放在tha3/app/emotions下
# 4. 本程序是一个整体，不可以单独运行某一步，因为每一步都需要上一步的输出作为输入,除非你自己将图片放在对应的目录下，并且按照要求修改好图片的名字
# 5. stable diffusion需要配置好inpainting模型，放在models/stable diffuison目录下
# 6. 如果你在单独运行第4步时出现了错误，你应该将第一步和第四步一起运行，因为第一步会输出一个图像的坐标，第四步需要用到这个坐标
# 7. 如果你在使用自己的stable diffuison时，发生了路径错误，请将"D:\stable-diffusion-webui-master\repositories\stable-diffusion-stability-ai\ldm\modules\encoders\modules.py"文件中的99和299行的文件路径的开头部分进行修改

import sys

sys.path.append('D:\\FaceMind_AIGC\\CartoonSegmentation')
from get_mask import crop_image
from get_mask import save_masks
sys.path.append('D:\\FaceMind_AIGC\\APISR')
from test_code import inference
from test_code.test_utils import load_grl, load_rrdb, load_dat, load_cunet
import video_super_res.video_super_res as vsr

from torchvision import transforms
import argparse
import os
import sys
import threading
import time
from typing import Optional
sys.path.append(os.getcwd())
from tha3.mocap.ifacialmocap_v2 import IFACIALMOCAP_PORT, IFACIALMOCAP_START_STRING, parse_ifacialmocap_v2_pose, \
    parse_ifacialmocap_v1_pose
from tha3.poser.modes.load_poser import load_poser
import cv2
import torch
import wx
import mediapipe
from tha3.poser.poser import Poser
from tha3.mocap.ifacialmocap_constants import *
from tha3.mocap.mediapipe_constants import HEAD_ROTATIONS, HEAD_X, HEAD_Y, HEAD_Z
from tha3.mocap.mediapipe_face_pose import MediaPipeFacePose
from tha3.mocap.mediapipe_face_pose_converter_00 import MediaPoseFacePoseConverter00
from tha3.mocap.ifacialmocap_pose import create_default_ifacialmocap_pose
from tha3.mocap.ifacialmocap_v2 import IFACIALMOCAP_PORT, IFACIALMOCAP_START_STRING, parse_ifacialmocap_v2_pose, \
    parse_ifacialmocap_v1_pose
from tha3.poser.modes.load_poser import load_poser
from scipy.spatial.transform import Rotation
import torch
import wx
import time
from tha3.util import resize_PIL_image, extract_PIL_image_from_filelike, \
    extract_pytorch_image_from_PIL_image
# from image_util import convert_linear_to_srgb
from image_util2 import convert_linear_to_srgb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import jwt
import datetime
import numpy as np
import wave
import IPython.display as ipd
import os
from moviepy.editor import ImageSequenceClip, AudioFileClip
from IPython.display import HTML
from base64 import b64encode
import re
import time
import os
import shutil




class PoseFilter2:
    def __init__(self, window_size):
        self.window_size = window_size
        self.history_pose = []

    def update(self, pose):
        self.history_pose.append(pose)
        # print("历史参数列表：")
        # for row in self.history_pose:
        #     print(row)
        if len(self.history_pose) > self.window_size:
            self.history_pose.pop(0)
        ans=np.mean(self.history_pose, axis=0).tolist()
        # print("滤波后：\n",ans)
        return ans

class MainFrame2():

    def __init__(self, poser: Poser, pose_converter, device,video_capture,face_landmarker):
    
        self.pose_converter = pose_converter
        self.poser = poser
        self.device = device
        self.biyao=0
        self.video_capture = video_capture
        self.face_landmarker = face_landmarker
        self.ifacialmocap_pose = create_default_ifacialmocap_pose()
        self.wx_source_image = None
        self.torch_source_image = None
        self.last_pose = None
        self.last_update_time = None
        self.rotation_value_labels = {}
        self.mediapipe_face_pose = None
        self.rotation_value_labels[HEAD_X]=None
        self.rotation_value_labels[HEAD_Y]=None
        self.rotation_value_labels[HEAD_Z]=None
        # self.load_image('tha3/app/image_with_alpha.png')
        # self.update_capture_panel()
        # self.update_result_image_bitmap()


    def paint_capture_panel(self, event: wx.Event):
        self.update_capture_panel(event)

    def update_capture_panel(self):
        there_is_frame, frame = self.video_capture.read()
        # rgb_frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not there_is_frame:
            print("Failed to grab a frame")
    # 处理失败情况，例如通过continue跳过当前循环迭代，或者根据具体情况处理
        else:
            print('sucess')
    # 如果成功读取帧，继续进行颜色转换和翻转操作
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (256, 192))
        time_ms = int(time.time() * 1000)
        mediapipe_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)
        cv2.imshow('RGB Frame', rgb_frame)

    def update_capture_frame(self,frame):
        frame = frame
        rgb_frame =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 如果成功读取帧，继续进行颜色转换和翻转操作
 
        resized_frame = cv2.resize(rgb_frame, (256, 192))
        time_ms = int(time.time() * 1000)
        mediapipe_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.face_landmarker.detect_for_video(mediapipe_image, time_ms)


        self.update_mediapipe_face_pose(detection_result)
       
    @staticmethod
    def convert_to_100(x):
        return int(max(0.0, min(1.0, x)) * 100)


    def update_mediapipe_face_pose(self, detection_result):
        self.biyao=len(detection_result.facial_transformation_matrixes)
        if len(detection_result.facial_transformation_matrixes) == 0:
            print(2)
            return
        xform_matrix = detection_result.facial_transformation_matrixes[0]
        blendshape_params = {}
        for item in detection_result.face_blendshapes[0]:
            blendshape_params[item.category_name] = item.score
        M = xform_matrix[0:3, 0:3]
        rot = Rotation.from_matrix(M)
        euler_angles = rot.as_euler('xyz', degrees=True)

        self.mediapipe_face_pose = MediaPipeFacePose(blendshape_params, xform_matrix)

    def update_result_image_bitmap(self,eyebrow_filter,eyes_filter,body_filter,mode,strength):
        if self.biyao==0:
             return
        current_pose = self.pose_converter.convert(self.mediapipe_face_pose)#表示视频当前帧的面部姿势
        #眉毛0~11,使用中等强度的滤波器
        # current_pose[0:12]=eyebrow_filter.update(current_pose[0:12])
        #眼睛12~25，使用最弱的滤波器，防止眼睛不变
        current_pose[12:26]=eyes_filter.update(current_pose[12:26])
         #嘴巴26~36，使用输入的参数控制
        if mode=='a'or mode=='i'or mode=='u'or mode=='e'or mode=='o' or mode=='smirk'or mode=='delta':
            if mode=='a':
                current_pose[26]=strength
            elif mode=='i':
                current_pose[27]=strength
            elif mode=='u':
                current_pose[28]=strength
            elif mode=='e':
                current_pose[29]=strength
            elif mode=='o':
                current_pose[30]=strength
            elif mode=='delta':
                current_pose[31]=strength
            elif mode=='smirk':
                current_pose[36]=strength
        elif mode=='b':
            for i in range(26,37):
                current_pose[i]=0   
        else:
            if mode=='lower':
                current_pose[32:34]=[strength,strength]
            elif mode=='raise':
                current_pose[34:36]=[strength,strength]
        
            
        #对身体采用滤波器37~44        
        # print("滤波前：\n",current_pose)
        current_pose[37:45] = body_filter.update(current_pose[37:45])#根据历史参数列表进行滤波  
        pose = torch.tensor(current_pose, device=self.device, dtype=self.poser.get_dtype())

        with torch.no_grad():
            output_image = self.poser.pose(self.torch_source_image, pose)[0].float()
            output_image = torch.clip((output_image + 1.0) / 2, 0.0, 1.0)
            output_image = convert_linear_to_srgb(output_image)     

            background_choice = 0
            if background_choice == 0:#透明背景
                pass
            else:
                background = torch.zeros(4, output_image.shape[1], output_image.shape[2], device=self.device)
                background[3, :, :] = 1.0
                if background_choice == 1:#绿色背景
                    background[1, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 2:#蓝色背景
                    background[2, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 3:#黑色背景
                    output_image = self.blend_with_background(output_image, background)
                else:#白色背景
                    background[0:3, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)

            c, h, w = output_image.shape
            output_image = 255* torch.transpose(output_image.reshape(c, h * w), 0, 1).reshape(h, w, c)
            output_image = output_image.byte()
        numpy_image = output_image.detach().cpu().numpy()        

        alpha_channel = numpy_image[:,:,3]
        rgb_image=cv2.cvtColor(numpy_image[:,:,:3],cv2.COLOR_BGRA2RGB)# 将图像转换为 RGB 格式（不包括透明通道）
        rgba_image=cv2.merge((rgb_image,alpha_channel)) # 将透明通道添加回 RGB 图像中
        return rgba_image
    def blend_with_background(self, numpy_image, background):
        alpha = numpy_image[3:4, :, :]
        color = numpy_image[0:3, :, :]
        new_color = color * alpha + (1.0 - alpha) * background[0:3, :, :]
        return torch.cat([new_color, background[3:4, :, :]], dim=0)

    def load_image(self,file_address):
            image_file_name =file_address
            try:
                pil_image = resize_PIL_image(
                    extract_PIL_image_from_filelike(image_file_name),
                    (self.poser.get_image_size(), self.poser.get_image_size()))
                w, h = pil_image.size
                if pil_image.mode != 'RGBA':
                    self.source_image_string = "Image must have alpha channel!"
                    self.wx_source_image = None
                    self.torch_source_image = None
                else:
                    self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image) \
                        .to(self.device).to(self.poser.get_dtype())
                    print('successfully load')
            except:
                print('加载图片出现错误')
    def load_image1(self,file_address):
                image_file_name =file_address
                pil_image = resize_PIL_image(
                    extract_PIL_image_from_filelike(image_file_name),
                    (self.poser.get_image_size(), self.poser.get_image_size()))
                if pil_image.mode != 'RGBA':
                    self.source_image_string = "Image must have alpha channel!"
                    self.wx_source_image = None
                    self.torch_source_image = None
                else:
                    self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image) \
                        .to(self.device).to(self.poser.get_dtype())
                    print('successfully load')




class units2():
    def __init__(self):
        self.left=0
        self.top=0
        self.left_in_square=0
        self.top_in_square=0
        self.mask_width=0
        self.mask_height=0

    def downsample_frames(self,frames, original_fps, target_fps):
        # Calculate the ratio of the original fps to the target fps
        ratio = original_fps / target_fps
        
        # Select frames based on the ratio
        downsampled_frames = [frames[int(i * ratio)] for i in range(int(len(frames) / ratio))]
        
        return downsampled_frames

    def get_video_fps(self,video_path):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps  
    def step1(self,original_image_path,mask_output_dir):
        cropped_path=crop_image(original_image_path)
        print(cropped_path)
        time.sleep(3)
        #self.left,self.top,self.left_in_square,self.top_in_square,self.mask_width,self.mask_height=save_masks(cropped_path,mask_output_dir,cropped_dir)
        self.left,self.top,self.left_in_square,self.top_in_square,self.mask_width,self.mask_height=save_masks(original_image_path,mask_output_dir,cropped_dir)
    def step2(self,original_image_path,mask_output_path,background_path):
        #将遮罩图片和原始图片输入sd，重绘得到背景图片
        import requests
        import subprocess
        import io
        import base64
        from PIL import Image
        #diffusion_process=subprocess.Popen(['D:/stable-diffusion-webui-master/webui.bat','--api'])
        time.sleep(5)
        img_width=0
        img_height=0
        #把图片和mask转换成base64编码，如果图片太大，进行降采样
        with open(original_image_path, "rb") as f:
            original_image = f.read()
            img_width=Image.open(io.BytesIO(original_image)).size[0]
            img_height=Image.open(io.BytesIO(original_image)).size[1]
            new_width=img_width
            new_height=img_height
            while(new_width*new_height>1200*1200):
                original_image_PIL=Image.open(io.BytesIO(original_image))
                new_width=new_width//2
                new_height=new_height//2
                original_image_PIL=original_image_PIL.resize((new_width,new_height),resample=Image.BICUBIC)
                byte_arr = io.BytesIO()
                original_image_PIL.save(byte_arr, format='PNG')
                original_image = byte_arr.getvalue()
            # with open('outputoriginal.png', 'wb') as f:
            #     f.write(original_image)
            original_base64 = base64.b64encode(original_image).decode()
        with open(mask_output_path, "rb") as f:
            mask_image = f.read()
            if(img_height*img_width>1200*1200):
                mask_image = Image.open(io.BytesIO(mask_image)).resize((new_width,new_height),resample=Image.BICUBIC)
                byte_arr = io.BytesIO()
                mask_image.save(byte_arr, format='PNG')
                mask_image = byte_arr.getvalue()
            # with open('outputmask.png', 'wb') as f:
            #     f.write(mask_image)
            mask_base64 = base64.b64encode(mask_image).decode()
        url = "http://127.0.0.1:7860"         
        payload = {
            "prompt":"",
            "negative_prompt":"1girl,human",
            "override_settings":{
                "sd_model_checkpoint":"sd-v1-5-inpainting",
                "sd_vae":"Automatic"
            },
            "seed":-1,
            "batch_size":1,
            "n_iter":1,
            "steps":20,
            "cfg_scale":7,
            "width":new_width,
            "height":new_height,
            "resotre_faces":False,
            "tiling":False,
            "eta":0,
            "script_args":[],
            "sampler_index":"Euler a",
            "init_images":[original_base64],
            "mask":mask_base64,
            "resize_mode":1,
            "denoising_strength":1,
            "mask_blur":0,
            "mask_mode":0,    
        }
        while True:
            try:
                response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
                break
            except:
                print("sd启动中，请等待")
        # response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
        r = response.json()
        print(r)
        for i in r['images']:
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
            #如果之前降采样了，现在要恢复原来的大小
            if(img_width*img_height>1200*1200):
                image=image.resize((img_width,img_height),resample=Image.BICUBIC)
            image.save(background_path)    
            print("\nbackground image saved")
        #关闭sd进程
        # diffusion_process.terminate()
        # if diffusion_process.poll() is None:
        #     diffusion_process.kill()
        # print("sd process killed")

    def step3(self,input_image_path,emotion_video_path,background_path):
        img=Image.open(input_image_path)
        image_width,image_height=img.size
        parser = argparse.ArgumentParser(description='Control characters with movement captured by iFacialMocap.')
        parser.add_argument(
                '--model',
                type=str,
                required=False,
                default='standard_float',
                choices=['standard_float', 'separable_float', 'standard_half', 'separable_half'],
                help='The model to use.')
        args = parser.parse_args()
        device = torch.device('cuda')
        try:
                poser = load_poser(args.model, device)#加载模型
        except RuntimeError as e:
                print(e)
                sys.exit()
        device = torch.device("cuda:0")
        pose_converter = MediaPoseFacePoseConverter00()
        face_landmarker_base_options = mediapipe.tasks.BaseOptions(
                model_asset_path='data/face_landmarker_v2_with_blendshapes.task')#加载人脸标记器
        options = mediapipe.tasks.vision.FaceLandmarkerOptions(
                base_options=face_landmarker_base_options,
                running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1)



        # transform_mode=['a','i','o']
        transform_mode=['a','e','i','o','u']
        transform_strength=[1.0,0.75,0.5,0.25]
        # transform_strength=[1,0.5,0.25]
        for mode in transform_mode:
            print("###################################################开始合成%s表情###################################################"%mode)
            if  mode=='a'or mode=='i'or mode=='u'or mode=='e'or mode=='o' :
              for strength in transform_strength:
                print("###################################################开始合成%s表情的强度为%f###################################################"%(mode,strength))
                face_landmarker = mediapipe.tasks.vision.FaceLandmarker.create_from_options(options)
                video_capture = cv2.VideoCapture(emotion_video_path)
                main_frame = MainFrame2(poser,pose_converter, device,video_capture,face_landmarker)
                main_frame.load_image1(input_image_path)
                eyebrow_filter=PoseFilter2(12)#眉毛滤波器
                eyes_filter = PoseFilter2(1)#眼睛滤波器
                body_filter = PoseFilter2(15)#身体滤波器
                img_num=0
                cap = cv2.VideoCapture(emotion_video_path)
                while cap.isOpened() and img_num<100:
                    img_num+=1
                    ret, frame = cap.read()
                    if not ret:
                        break
                    main_frame.update_capture_frame(frame)

                    numpy_image=main_frame.update_result_image_bitmap(eyebrow_filter,eyes_filter,body_filter,mode,strength)#嘴巴mode放在这里
                    if numpy_image is None:
                        continue
                    resized_image = cv2.resize(numpy_image, (image_width, image_height), interpolation=cv2.INTER_LANCZOS4)
                    resized_image=cv2.cvtColor(resized_image,cv2.COLOR_BGRA2RGBA)
                    resized_image=Image.fromarray(resized_image)#PIL图像  
                    resized_image.save('imageout1.png')
                    

                    #保存路径
                    savepath='C:/Users/FM/Desktop/facial/beishang2/'+mode+'_'+str(strength)+'/img_%d.webp' % img_num
                    if not os.path.exists('C:/Users/FM/Desktop/facial/beishang2/'+mode+'_'+str(strength)):
                        os.makedirs('C:/Users/FM/Desktop/facial/beishang2/'+mode+'_'+str(strength))
                    #把透明人物拼接到背景上，输出的是pil格式
                    resized_image=self.step4(Image.open(background_path),resized_image)
                    #不超分（二选一）
                    resized_image.save(savepath)
                    print("saving to:",savepath)
                    #超分（二选一）
                    # resized_image=np.array(resized_image)
                    # resized_image=psr.super_resolve_video("D:\\FaceMind_AIGC\\APISR\\pretrained\\4x_APISR_RRDB_GAN_generator.pth",resized_image)
                    # resized_image = resized_image.squeeze(0)
                    # to_pil=transforms.ToPILImage()                    
                    # pil_img=to_pil(resized_image)
                    # b,g,r=pil_img.split()
                    # pil_img=Image.merge("RGB",(r,g,b))
                    # pil_img.save(savepath)
                    # print("saving to:",savepath)

                cap.release()
            else:
                strength=0.0
                strength2=1
                print("###################################################开始合成%s表情的强度为%f###################################################"%(mode,strength))
                face_landmarker = mediapipe.tasks.vision.FaceLandmarker.create_from_options(options)
                video_capture = cv2.VideoCapture(emotion_video_path)
                main_frame = MainFrame2(poser,pose_converter, device,video_capture,face_landmarker)
                main_frame.load_image1(input_image_path)
                eyebrow_filter=PoseFilter2(12)#眉毛滤波器
                eyes_filter = PoseFilter2(1)#眼睛滤波器
                body_filter = PoseFilter2(15)#身体滤波器
                img_num=0
                cap = cv2.VideoCapture(emotion_video_path)
                while cap.isOpened() and img_num<100:
                    img_num+=1
                    ret, frame = cap.read()
                    if not ret:
                        break
                    main_frame.update_capture_frame(frame)

                    numpy_image=main_frame.update_result_image_bitmap(eyebrow_filter,eyes_filter,body_filter,mode,strength)#嘴巴mode放在这里
                    if numpy_image is None:
                        continue
                    resized_image = cv2.resize(numpy_image, (image_width,image_height))#numpy图像
                    resized_image=cv2.cvtColor(resized_image,cv2.COLOR_BGRA2RGBA)
                    resized_image=Image.fromarray(resized_image)#PIL图像  
                    

                    #保存路径
                    savepath='C:/Users/FM/Desktop/facial/beishang2/'+mode+'_'+str(strength)+'/img_%d.webp' % img_num
                    if not os.path.exists('C:/Users/FM/Desktop/facial/beishang2/'+mode+'_'+str(strength)):
                        os.makedirs('C:/Users/FM/Desktop/facial/beishang2/'+mode+'_'+str(strength))
                    #把透明人物拼接到背景上，输出的是pil格式
                    resized_image=self.step4(Image.open(background_path),resized_image)
                    #不超分（二选一）
                    resized_image.save(savepath)
                    print("saving to:",savepath)

        print("###################################################拼接完成！###################################################")
        


    def step4(self,background,img): 
        left=self.left_in_square
        top=self.top_in_square
        right=self.left_in_square+self.mask_width
        bottom=self.top_in_square+self.mask_height 
        # 把人物图片从正方形裁剪出来
        img = img.crop((left, top, right, bottom))  
        # 把人物和背景的拼接结果存到new_img里            
        new_img = Image.new('RGBA', background.size)
        new_img.paste(background)             
        new_img.paste(img, (self.left, self.top), img)
        # 保存拼接结果
        return new_img




original_image_path='tha3/app/source_imgs/img_0.png'#原始图片的输入路径


#下面这些不用改
input_image_path='tha3/app/mask_output/img_0.png'#输入图像的路径(分割好之后不带背景的人物图片)
mask_output_dir='tha3/app/mask_output/'#没有背景的人物图像和遮罩的输出路径
background_output_video_path='tha3/app/background_output/output.mp4'#加上背景的视频输出路径
vsr_input_video_path=background_output_video_path#超分输入视频的路径(就是背景输出视频的路径)
vsr_output_video_path='tha3/app/final_output/output.mp4'#超分输出视频的路径

cropped_dir='tha3/app/cropped_dir/'
cropped_path=r"D:\FaceMind_AIGC\talking\tha3\app\source_imgs\cropped_image.png"

a2='嗔怒4.mp4'#参考表情视频的文件名（包含扩展名）
background_path='tha3/app/background/background.png'#用于合成的背景路径

emotion_video_path2 = 'tha3/app/emotions/'+a2#表情视频的路径
units_ins2=units2()
time1=time.time()
#step1:抠出人物图片和遮罩(source_imgs->mask_output)
units_ins2.step1(original_image_path,mask_output_dir)
print("----------------------------step1 finished!!-------------------------------------")
end_time1=time.time()


units_ins2.step2(original_image_path,mask_output_dir+'mask_0.png',background_path)
print("----------------------------step2 finished!!-------------------------------------")

time2=time.time()
#step3:输入人物图片+表情视频，输出透明背景的人物表情图片(emotions+mask_output->emotion_output)
#注意：step4已经被整合进了step3，为了提高合成速度，去掉了中间保存每一帧的图片的步骤。要注意，step3不能和step1分开运行，因为在拼接视频的时候，用到了step1的输出结果
units_ins2.step3(input_image_path,emotion_video_path2,background_path)
print("----------------------------step3 finished!!-------------------------------------")
end_time2=time.time()
print("step1 time:",end_time1-time1,"s")
print("step2 time:",end_time2-time2,"s")




