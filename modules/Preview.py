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
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('.')+'/lib')
sys.path.append(os.path.abspath('.')+'/lib/CartoonSegmentation/')
#print(os.path.abspath('.')+'/CartoonSegmentation/')
sys.path.append(os.path.abspath('.')+'/lib/APISR')
sys.path.append(os.path.abspath('.')+'/lib/talking/')
from lib.CartoonSegmentation.get_mask import crop_image
from lib.CartoonSegmentation.get_mask import save_masks

import lib.talking.tha3.app.video_super_res.video_super_res as vsr
from moviepy.editor import *
from torchvision import transforms
import argparse
import sys
import threading
import time
from typing import Optional
#_1 sys.path.append(os.getcwd())
from lib.talking.tha3.mocap.ifacialmocap_v2 import IFACIALMOCAP_PORT, IFACIALMOCAP_START_STRING, parse_ifacialmocap_v2_pose, \
    parse_ifacialmocap_v1_pose
from lib.talking.tha3.poser.modes.load_poser import load_poser
import cv2
import torch
import wx
import mediapipe
from lib.talking.tha3.poser.poser import Poser
from lib.talking.tha3.mocap.ifacialmocap_constants import *
from lib.talking.tha3.mocap.mediapipe_constants import HEAD_ROTATIONS, HEAD_X, HEAD_Y, HEAD_Z
from lib.talking.tha3.mocap.mediapipe_face_pose import MediaPipeFacePose
from lib.talking.tha3.mocap.mediapipe_face_pose_converter_00 import MediaPoseFacePoseConverter00
from lib.talking.tha3.mocap.ifacialmocap_pose import create_default_ifacialmocap_pose
from lib.talking.tha3.mocap.ifacialmocap_v2 import IFACIALMOCAP_PORT, IFACIALMOCAP_START_STRING, parse_ifacialmocap_v2_pose, \
    parse_ifacialmocap_v1_pose
from lib.talking.tha3.poser.modes.load_poser import load_poser
from scipy.spatial.transform import Rotation
import torch
import wx
import time
from lib.talking.tha3.util import resize_PIL_image, extract_PIL_image_from_filelike, \
    extract_pytorch_image_from_PIL_image
# from image_util import convert_linear_to_srgb
from modules.image_util2 import convert_linear_to_srgb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import jwt
import datetime
import wave
import IPython.display as ipd
from moviepy.editor import ImageSequenceClip, AudioFileClip
from IPython.display import HTML
from base64 import b64encode
import re
import shutil
# Function to generate a JWT token
def generate_token(username: str="dev") -> str:
    AUTH_SECRET_KEY = "j45s2drxb1a0l9twsg8byd8765xerb751ttoyce5tavqlqwo2gur9hpyumcgk3v7"
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=72)
    }
    token = "Bearer " + jwt.encode(payload, AUTH_SECRET_KEY, algorithm="HS256")
    return token

# Generate the token

#new_token = generate_token()

# Set up the API request
#url = "http://otaku.facemind.wiki:5101/synthesize/"
#payload = json.dumps({
#    "ref_ids": ["9179cf38-12c6-11ef-9379-0b155492e0b3"],
#    "fuse_ref_rate": [1.0],
#    "text": "公爵？没想到这个时间你会到医务室来，哎呀，应该没有受伤吧？",
#    "role_base": "jixuanyou",
#    "return_phone": True,
#})
#headers = {
#    'Authorization': new_token,
#    'Content-Type': 'application/json'
#}
#哈喽，很高兴认识你呀，我是魈，你叫什么名字呢？
#让我想想这道题应该怎么做呢，奥对了，可以用那个方法，谢谢你！
#哎呀，你不要这样，讨厌!

"""url = "http://otaku.facemind.wiki:5101/synthesize/"
payload = json.dumps({
            "text": "大家好呀。我是花火哦，我可是一个出自米哈游的星穹铁道的角色呢。",
            "bot":"花火",
            "role_base": "huahuo",
            "ref_id": "35a2ce6c-2ee1-11ef-9461-cd5e4f2702a1",
          "return_phone": True,
        })
headers = {
    'Authorization': new_token,
    'Content-Type': 'application/json'
}"""

def unpack_bytes(data):
    pattern = re.compile(rb';;\[.*?\]')
    matches = pattern.finditer(data)

    pcm_chunks = []
    mouth_shape_sequences = []
    last_index = 0

    for match in matches:
        split_index = match.start()

        # PCM data before the match
        pcm_bytes = data[last_index:split_index]
        pcm_chunks.append(pcm_bytes)

        # Mouth shape sequence
        mouth_shape_bytes = match.group()
        mouth_shape_sequences.append(mouth_shape_bytes)

        last_index = match.end()

    # Process remaining PCM data after the last match
    if last_index < len(data):
        pcm_chunks.append(data[last_index:])

    # Combine all PCM data chunks
    combined_pcm_bytes = b''.join(pcm_chunks)

    # Convert to int16 array
    pcm_data = np.frombuffer(combined_pcm_bytes, dtype=np.int16)

    return pcm_data, mouth_shape_sequences

def adjust_elements(lst):
    def extract_value(s):
        return float(s.split('_')[1])

    def create_element(prefix, value):
        return f"{prefix}_{value}"

    n = len(lst)
    for i in range(n - 1):
        current_prefix = lst[i].split('_')[0]
        next_prefix = lst[i + 1].split('_')[0]

        # Skip adjustment if the current or next element starts with 'b'
        if current_prefix == 'b' or next_prefix == 'b':
            continue

        current_value = extract_value(lst[i])
        next_value = extract_value(lst[i + 1])
        if abs(current_value - next_value) > 0.25:
            if current_value < next_value:
                lst[i] = create_element(current_prefix, next_value - 0.25)
            else:
                lst[i + 1] = create_element(next_prefix, current_value - 0.25)

    return lst

def create_looping_frames(max_frame, length):
    sequence = list(range(1, max_frame + 1)) + list(range(max_frame, 1, -1))
    repeating_sequence = []
    while len(repeating_sequence) < length:
        repeating_sequence.extend(sequence)
    return repeating_sequence[:length]


def display_video(path):
    mp4 = open(path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f"""
    <video width=600 controls>
          <source src="{data_url}" type="video/mp4">
    </video>
    """)

def undersample_sequence(sequence, target_length):
    indices = np.linspace(0, len(sequence) - 1, target_length).astype(int)
    return [sequence[i] for i in indices]


class PoseFilter:
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

class MainFrame():

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

        #print(detection_result)
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
        #print(self.mediapipe_face_pose)

    def process_emotion_pose_pre_frame(self,eyebrow_filter,eyes_filter,body_filter,mouth_param,a,pose_tensor_list):
        if self.biyao==0:
             return
        # print(self.pose_converter)
        current_pose = self.pose_converter.convert(self.mediapipe_face_pose)#表示视频当前帧的面部姿势
        # print(current_pose)
        #眉毛0~11,使用中等强度的滤波器
        current_pose[0:12]=eyebrow_filter.update(current_pose[0:12])
        #眼睛12~25，使用最弱的滤波器，防止眼睛不变
        current_pose[12:26]=eyes_filter.update(current_pose[12:26])
        #嘴巴26~36，使用输入的参数控制
        if mouth_param=='b':
            for i in range(26,37):
                current_pose[i]=0
        else:
            parts=mouth_param.split('_')
            mouth_type=parts[0]
            strength=float(parts[1])
            if mouth_type=='a':
                current_pose[26]=strength
                for i in range(26,37):
                    if i!=26:
                        current_pose[i]=0
            elif mouth_type=='i':
                current_pose[27]=strength
                for i in range(26,37):
                    if i!=27:
                        current_pose[i]=0                
            elif mouth_type=='u':
                current_pose[28]=strength
                for i in range(26,37):
                    if i!=28:
                        current_pose[i]=0
            elif mouth_type=='e':
                current_pose[29]=strength
                for i in range(26,37):
                    if i!=29:
                        current_pose[i]=0
            elif mouth_type=='o':
                current_pose[30]=strength
                for i in range(26,37):
                    if i!=30:
                        current_pose[i]=0

        #对身体采用滤波器37~44        
        # print("滤波前：\n",current_pose)
        current_pose[37:45] = body_filter.update(current_pose[37:45])#根据历史参数列表进行滤波
        for i in range(39,42):
          if current_pose[i]<0:
            current_pose[i]=current_pose[i]-0.02
            if current_pose[i]<=-1:
                current_pose[i]=-1
          else:
            current_pose[i]=current_pose[i]+0.02
            if current_pose[i]>=1:
                current_pose[i]=1
        current_pose[42]=a
        print(a)
        for i in range(44,45):
          if current_pose[i]<0:
            current_pose[i]=current_pose[i]-0.02
            if current_pose[i]<=-1:
                current_pose[i]=-1
          else:
            current_pose[i]=current_pose[i]+0.02
            if current_pose[i]>=1:
                current_pose[i]=1
        # print(current_pose) 
        # print(len(current_pose))
        # print(current_pose[38]) 
        pose = torch.tensor(current_pose, device=self.device, dtype=self.poser.get_dtype())
        # print(pose)
        pose_tensor_list.append(pose)
        
        return pose_tensor_list
    
    def apply_emotion_pose(self,pose):
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

    

    def update_result_image_bitmap(self,eyebrow_filter,eyes_filter,body_filter,mouth_param,a):
        if self.biyao==0:
             return
        # print(self.pose_converter)
        current_pose = self.pose_converter.convert(self.mediapipe_face_pose)#表示视频当前帧的面部姿势
        # print(current_pose)
        #眉毛0~11,使用中等强度的滤波器
        current_pose[0:12]=eyebrow_filter.update(current_pose[0:12])
        #眼睛12~25，使用最弱的滤波器，防止眼睛不变
        current_pose[12:26]=eyes_filter.update(current_pose[12:26])
        #嘴巴26~36，使用输入的参数控制
        if mouth_param=='b':
            for i in range(26,37):
                current_pose[i]=0
        else:
            parts=mouth_param.split('_')
            mouth_type=parts[0]
            strength=float(parts[1])
            if mouth_type=='a':
                current_pose[26]=strength
                for i in range(26,37):
                    if i!=26:
                        current_pose[i]=0
            elif mouth_type=='i':
                current_pose[27]=strength
                for i in range(26,37):
                    if i!=27:
                        current_pose[i]=0                
            elif mouth_type=='u':
                current_pose[28]=strength
                for i in range(26,37):
                    if i!=28:
                        current_pose[i]=0
            elif mouth_type=='e':
                current_pose[29]=strength
                for i in range(26,37):
                    if i!=29:
                        current_pose[i]=0
            elif mouth_type=='o':
                current_pose[30]=strength
                for i in range(26,37):
                    if i!=30:
                        current_pose[i]=0

        #对身体采用滤波器37~44        
        # print("滤波前：\n",current_pose)
        current_pose[37:45] = body_filter.update(current_pose[37:45])#根据历史参数列表进行滤波
        for i in range(39,42):
          if current_pose[i]<0:
            current_pose[i]=current_pose[i]-0.02
            if current_pose[i]<=-1:
                current_pose[i]=-1
          else:
            current_pose[i]=current_pose[i]+0.02
            if current_pose[i]>=1:
                current_pose[i]=1
        current_pose[42]=a
        print(a)
        for i in range(44,45):
          if current_pose[i]<0:
            current_pose[i]=current_pose[i]-0.02
            if current_pose[i]<=-1:
                current_pose[i]=-1
          else:
            current_pose[i]=current_pose[i]+0.02
            if current_pose[i]>=1:
                current_pose[i]=1
        # print(current_pose) 
        # print(len(current_pose))
        # print(current_pose[38]) 
        pose = torch.tensor(current_pose, device=self.device, dtype=self.poser.get_dtype())
        #print(pose)
        
        """if not os.path.exists('/data/pose'):
            # 如果不存在，则创建目录
            os.makedirs('/data/pose')
        torch.save(pose, '/data/pose/pose.pt')
        pose = torch.load('/data/pose/pose.pt')"""

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




class units():
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
    def step1(self,original_image_path,mask_output_dir,cropped_dir):
        cropped_path=crop_image(original_image_path,cropped_dir)
        print(cropped_path)
        #time.sleep(3)
        #self.left,self.top,self.left_in_square,self.top_in_square,self.mask_width,self.mask_height=save_masks(cropped_path,mask_output_dir,cropped_dir)
        self.left,self.top,self.left_in_square,self.top_in_square,self.mask_width,self.mask_height=save_masks(original_image_path,mask_output_dir,cropped_dir)
    def step2(self,original_image_path,mask_output_path,background_path,is_trans = False, alternate_background = False):
        #将遮罩图片和原始图片输入sd，重绘得到背景图片
        import requests
        import subprocess
        import io
        import base64
        from PIL import Image
        #diffusion_process=subprocess.Popen(['D:/stable-diffusion-webui-master/webui.bat','--api'])
        """diffusion_process = subprocess.Popen(
            ['D:/AIGC/绘世启动器/sd-webui-aki-v4.9/python/python.exe', 
            'D:/AIGC/绘世启动器/sd-webui-aki-v4.9/launch.py', 
            '--api'],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )"""
        #time.sleep(5)
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

        if is_trans == False and alternate_background == False:
            url = "http://127.0.0.1:8848"         
            payload = {
                "prompt":"best quality,the background is the clean,",
                "negative_prompt":"1girl,human",
                "override_settings":{
                    "sd_model_checkpoint":"/Violet/Mix_Beta/Violet_mix_beta_002.safetensors",
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
                    print(response.status_code)  # 打印状态码
                    print(response.text)         # 打印响应内容
                    break
                except:
                    print("服务器未启,请尝试启动Webui进程, http://127.0.0.1:8848 ")
                    break
            # response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
            r = response.json()
            #print(r)
            for i in r['images']:
                image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
                
                #如果之前降采样了，现在要恢复原来的大小
                if(img_width*img_height>1200*1200):
                    image=image.resize((img_width,img_height),resample=Image.BICUBIC)
                image.save(background_path)    
                print("\nbackground image saved")
            #关闭sd进程
            """diffusion_process.terminate()
            if diffusion_process.poll() is None:
                diffusion_process.kill()"""
            #print("sd process killed")
        #else:
        #    image = Image.open(background_path)
        #    if(img_width*img_height>1200*1200):
        #        image=image.resize((img_width,img_height),resample=Image.BICUBIC)
        #    image.save(background_path)  
    
    def process_and_save_emotion_pose(self,emotion_video_path,background_output_video_path,face_landmarker_path, emotion_pose_save_path):
        parser = argparse.ArgumentParser(description='Control characters with movement captured by iFacialMocap.')
        parser.add_argument(
                '--model',
                type=str,
                required=False,
                default='standard_float',
                choices=['standard_float', 'separable_float', 'standard_half', 'separable_half'],
                help='The model to use.')
        #parser.add_argument('--config', default="./config/sitting.yaml", type=str, required=False, help='Path to the config file.')
        #parser.add_argument('--pipeline', type=str, default="PIFDFSFRF",required=False, help='Pipeline.')
        #parser.add_argument('--is_trans', type=str, default="None", required=False, help='Generate a transparent no background image package.')
        #parser.add_argument('--package_name', type=str, default=None, required=False, help='Output package name.')
        args, unknown_args = parser.parse_known_args()
        device = torch.device('cuda')
        try:
                poser = load_poser(args.model, device)#加载模型
                print(poser)
        except RuntimeError as e:
                print(e)
                sys.exit()
        device = torch.device("cuda:0")
        pose_converter = MediaPoseFacePoseConverter00()
        # print(pose_converter)
        print(pose_converter.head_x_index)
        print(pose_converter.head_y_index)
        print(pose_converter.neck_z_index)
        print(pose_converter.body_y_index)
        print(pose_converter.body_z_index)
        print(pose_converter.breathing_index)
        face_landmarker_base_options = mediapipe.tasks.BaseOptions(
                model_asset_path = face_landmarker_path)#加载人脸标记器
        options = mediapipe.tasks.vision.FaceLandmarkerOptions(
                base_options=face_landmarker_base_options,
                running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1)
        face_landmarker = mediapipe.tasks.vision.FaceLandmarker.create_from_options(options)
        video_capture = cv2.VideoCapture(emotion_video_path)
        main_frame = MainFrame(poser,pose_converter, device,video_capture,face_landmarker)
        #main_frame.load_image1(input_image_path)

        # 打开视频文件
        cap = cv2.VideoCapture(emotion_video_path)
        # 逐帧读取视频
        img_num=0
        eyebrow_filter=PoseFilter(12)#眉毛滤波器
        eyes_filter = PoseFilter(1)#眼睛滤波器
        body_filter = PoseFilter(15)#身体滤波器
        fps=self.get_video_fps(emotion_video_path)

        new_token = generate_token()
        url = "http://otaku.facemind.wiki:5101/synthesize/"
        payload = json.dumps({
                    "text": "大家好呀。我是花火哦，我可是一个出自米哈游的星穹铁道的角色呢。",
                    "bot":"花火",
                    "role_base": "huahuo",
                    "ref_id": "35a2ce6c-2ee1-11ef-9461-cd5e4f2702a1",
                "return_phone": True,
                })
        headers = {
            'Authorization': new_token,
            'Content-Type': 'application/json'
        }

        start_time = time.time()
        response = requests.post(url, headers=headers, data=payload)
        response_content = response.content
        pcm_data, mouth_shape_sequences = unpack_bytes(response_content)
        combined_mouth_shape=[]
        for i, mouth_shape in enumerate(mouth_shape_sequences):
            mouth_shape=json.loads((mouth_shape.decode('utf-8')).replace(';;',''))
            print(f"Mouth shape sequence {i}: {mouth_shape}")
            print(len(mouth_shape))
            combined_mouth_shape = combined_mouth_shape + mouth_shape
        print(combined_mouth_shape)

        output_list = adjust_elements(combined_mouth_shape)
        print(output_list)
        end_time = time.time()
        print(f"Time taken to generate token: {end_time - start_time:.2f} seconds")
        output_audio_file = background_output_video_path + "output.mp3"
        with wave.open(output_audio_file, "w") as wf:
             wf.setnchannels(1)  # mono
             wf.setsampwidth(2)  # sample width in bytes
             wf.setframerate(32000)  # frame rate
             wf.writeframes(pcm_data.tobytes())

        ipd.display(ipd.Audio(output_audio_file))


        sequence = output_list

        mouth_params = undersample_sequence(sequence, int(len(sequence)/3.333))

        a=0
        k=1

        pose_tensor_list = []
        pose_tensor_list.append(fps)

        while cap.isOpened() and img_num<100:
            img_num+=1
            ret, frame = cap.read()
            
            if not ret:
                break
            main_frame.update_capture_frame(frame)
            #假如给出的嘴型参数比参考视频的帧数少，那么就用闭嘴补全
            if img_num<=len(mouth_params):
                mouth_param=mouth_params[img_num-1]
            else :
                mouth_param='b'
            if k==1:
                a=a+0.003
                if a>=0.1:
                    k=0
            if k==0:
                a=a-0.003
                if a<=-0.3:
                    k=1
            #numpy_image=main_frame.update_result_image_bitmap(eyebrow_filter,eyes_filter,body_filter,mouth_param,a)
            pose_tensor_list = main_frame.process_emotion_pose_pre_frame(eyebrow_filter,eyes_filter,body_filter,mouth_param,a,pose_tensor_list)
        cap.release()
        video_name = os.path.basename(emotion_video_path)
        emotion_pose_save_path = emotion_pose_save_path + video_name 
        if not os.path.exists(emotion_pose_save_path):
            # 如果不存在，则创建目录
            os.makedirs(emotion_pose_save_path)
        torch.save(pose_tensor_list, emotion_pose_save_path + f'/{video_name}_preview.pt')

    def load_emotion_pose_from_tensor_list(self, emotion_pose_load_path):
        video_name = os.path.basename(emotion_pose_load_path)
        emotion_pose_tensor_list = torch.load(emotion_pose_load_path + f'/{video_name}_preview.pt')
        return emotion_pose_tensor_list
    
    def step3_pose_from_tensor(self,input_image_path,emotion_pose_load_path,background_path,background_output_video_path,face_landmarker_path, is_trans=False):
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
        #parser.add_argument('--config', default="./config/sitting.yaml", type=str, required=False, help='Path to the config file.')
        #parser.add_argument('--pipeline', type=str, default="PIFDFSFRF",required=False, help='Pipeline.')
        #parser.add_argument('--is_trans', type=str, default="None", required=False, help='Generate a transparent no background image package.')
        #parser.add_argument('--package_name', type=str, default=None, required=False, help='Output package name.')
        args, unknown_args = parser.parse_known_args()
        device = torch.device('cuda')
        try:
                poser = load_poser(args.model, device)#加载模型
                print(poser)
        except RuntimeError as e:
                print(e)
                sys.exit()
        device = torch.device("cuda:0")
        pose_converter = MediaPoseFacePoseConverter00()
        # print(pose_converter)
        face_landmarker_base_options = mediapipe.tasks.BaseOptions(
                model_asset_path = face_landmarker_path)#加载人脸标记器
        options = mediapipe.tasks.vision.FaceLandmarkerOptions(
                base_options=face_landmarker_base_options,
                running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1)
        face_landmarker = mediapipe.tasks.vision.FaceLandmarker.create_from_options(options)
        video_capture = cv2.VideoCapture()

        main_frame = MainFrame(poser,pose_converter, device,video_capture,face_landmarker)
        main_frame.load_image1(input_image_path)
        emotion_pose_tensor_list = self.load_emotion_pose_from_tensor_list(emotion_pose_load_path)
        fps = emotion_pose_tensor_list[0]
        background=Image.open(background_path)
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        out=cv2.VideoWriter(background_output_video_path+'output.mp4',fourcc,fps,(background.size[0],background.size[1]))

        img_num = 0
        for pose in emotion_pose_tensor_list[1:]:
            img_num = img_num + 1
            numpy_image = main_frame.apply_emotion_pose(pose)
            if numpy_image is None:
                continue
            resized_image = cv2.resize(numpy_image, (image_width,image_height))#numpy图像
            resized_image = cv2.cvtColor(resized_image,cv2.COLOR_BGRA2RGBA)
            resized_image = Image.fromarray(resized_image)#PIL图像 

            #image = Image.fromarray(resized_image)
            #resized_image.save(r'C:\Users\Violet\Desktop\新建文件夹\output_image.png')

            self.step4(out,background,resized_image,is_trans)
            print('img_%d saved'%img_num)
        print("###################################################视频拼接完成！###################################################")
        out.release()


    def step3(self,input_image_path,emotion_video_path,background_path,background_output_video_path,face_landmarker_path, is_trans=False):
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
        #parser.add_argument('--config', default="./config/sitting.yaml", type=str, required=False, help='Path to the config file.')
        #parser.add_argument('--pipeline', type=str, default="PIFDFSFRF",required=False, help='Pipeline.')
        #parser.add_argument('--is_trans', type=str, default="None", required=False, help='Generate a transparent no background image package.')
        #parser.add_argument('--package_name', type=str, default=None, required=False, help='Output package name.')
        args, unknown_args = parser.parse_known_args()
        device = torch.device('cuda')
        try:
                poser = load_poser(args.model, device)#加载模型
                print(poser)
        except RuntimeError as e:
                print(e)
                sys.exit()
        device = torch.device("cuda:0")
        pose_converter = MediaPoseFacePoseConverter00()
        # print(pose_converter)
        print(pose_converter.head_x_index)
        print(pose_converter.head_y_index)
        print(pose_converter.neck_z_index)
        print(pose_converter.body_y_index)
        print(pose_converter.body_z_index)
        print(pose_converter.breathing_index)
        face_landmarker_base_options = mediapipe.tasks.BaseOptions(
                model_asset_path = face_landmarker_path)#加载人脸标记器
        options = mediapipe.tasks.vision.FaceLandmarkerOptions(
                base_options=face_landmarker_base_options,
                running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1)
        face_landmarker = mediapipe.tasks.vision.FaceLandmarker.create_from_options(options)
        video_capture = cv2.VideoCapture(emotion_video_path)
        main_frame = MainFrame(poser,pose_converter, device,video_capture,face_landmarker)
        main_frame.load_image1(input_image_path)

        # 打开视频文件
        cap = cv2.VideoCapture(emotion_video_path)
        # 逐帧读取视频
        img_num=0
        eyebrow_filter=PoseFilter(12)#眉毛滤波器
        eyes_filter = PoseFilter(1)#眼睛滤波器
        body_filter = PoseFilter(15)#身体滤波器

        fps=self.get_video_fps(emotion_video_path)
        background=Image.open(background_path)
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        out=cv2.VideoWriter(background_output_video_path+'output.mp4',fourcc,fps,(background.size[0],background.size[1]))
        #print(background_output_video_path,"################")

        # #从文本文件中读取嘴型参数
        # with open(r'tha3\app\mouth_params.txt') as f:
        #     params=f.read()
        # params=params.split(',')
        # params=[param.strip() for param in params]
        # mouth_params=[param.strip("'") for param in params]

        # print("降采样前：",len(mouth_params))
        # mouth_params=self.downsample_frames(mouth_params,150,fps)#从原来的100帧降到目标帧率
        # print("降采样后：",len(mouth_params))

        new_token = generate_token()
        url = "http://otaku.facemind.wiki:5101/synthesize/"
        payload = json.dumps({
                    "text": "大家好呀。我是花火哦，我可是一个出自米哈游的星穹铁道的角色呢。",
                    "bot":"花火",
                    "role_base": "huahuo",
                    "ref_id": "35a2ce6c-2ee1-11ef-9461-cd5e4f2702a1",
                "return_phone": True,
                })
        headers = {
            'Authorization': new_token,
            'Content-Type': 'application/json'
        }

        start_time = time.time()
        response = requests.post(url, headers=headers, data=payload)
        response_content = response.content
        pcm_data, mouth_shape_sequences = unpack_bytes(response_content)
        combined_mouth_shape=[]
        for i, mouth_shape in enumerate(mouth_shape_sequences):
            mouth_shape=json.loads((mouth_shape.decode('utf-8')).replace(';;',''))
            print(f"Mouth shape sequence {i}: {mouth_shape}")
            print(len(mouth_shape))
            combined_mouth_shape = combined_mouth_shape + mouth_shape
        print(combined_mouth_shape)

        output_list = adjust_elements(combined_mouth_shape)
        print(output_list)
        end_time = time.time()
        print(f"Time taken to generate token: {end_time - start_time:.2f} seconds")
        output_audio_file = background_output_video_path + "output.mp3"
        with wave.open(output_audio_file, "w") as wf:
             wf.setnchannels(1)  # mono
             wf.setsampwidth(2)  # sample width in bytes
             wf.setframerate(32000)  # frame rate
             wf.writeframes(pcm_data.tobytes())

        ipd.display(ipd.Audio(output_audio_file))

        # path = "C:/Users/FM/Desktop/facial/huohua"
        sequence = output_list
        # special_mapping = {
        #         "b_0.0": "b_1"
    
        #             }

        #if not os.path.exists('frames'):
                #os.makedirs('frames')

        # Undersample the mouth shape sequence from 100 to 30 frames
        mouth_params = undersample_sequence(sequence, int(len(sequence)/3.333))
        #required_length = len(undersampled_sequence)
        #looping_frames = create_looping_frames(100, required_length)
        a=0
        k=1
        while cap.isOpened() and img_num<100:
            img_num+=1
            ret, frame = cap.read()
            #image = Image.fromarray(frame)
            #image.save(r'C:\Users\Violet\Desktop\新建文件夹\output_frame.png')
            
            if not ret:
                break
            main_frame.update_capture_frame(frame)
            #假如给出的嘴型参数比参考视频的帧数少，那么就用闭嘴补全
            if img_num<=len(mouth_params):
                mouth_param=mouth_params[img_num-1]
            else :
                mouth_param='b'
            if k==1:
                a=a+0.003
                if a>=0.1:
                    k=0
            if k==0:
                a=a-0.003
                if a<=-0.3:
                    k=1
            numpy_image=main_frame.update_result_image_bitmap(eyebrow_filter,eyes_filter,body_filter,mouth_param,a)

            if numpy_image is None:
                continue
            resized_image = cv2.resize(numpy_image, (image_width,image_height))#numpy图像
            resized_image=cv2.cvtColor(resized_image,cv2.COLOR_BGRA2RGBA)
            resized_image=Image.fromarray(resized_image)#PIL图像 

            #image = Image.fromarray(resized_image)
            #resized_image.save(r'C:\Users\Violet\Desktop\新建文件夹\output_image.png')
            self.step4(out,background,resized_image,is_trans)
            print('img_%d saved'%img_num)
        print("###################################################视频拼接完成！###################################################")
        cap.release()
        out.release()


    def step4(self,out,background,img,is_trans=False): 

        #把人物图片从正方形裁剪出来
        left=self.left_in_square
        top=self.top_in_square
        right=self.left_in_square+self.mask_width
        bottom=self.top_in_square+self.mask_height 

        img = img.crop((left, top, right, bottom))  # 把人物图片从正方形裁剪出来
        # 把人物和背景的拼接结果存到new_img里            
        new_img = Image.new('RGBA', background.size)
        if is_trans==False: new_img.paste(background)             
        new_img.paste(img, (self.left, self.top), img)
             
        # 保存拼接结果
        new_img_array = np.array(new_img)  # 将PIL图像转换为NumPy数组
        new_img_array = cv2.cvtColor(new_img_array, cv2.COLOR_RGBA2BGR)  # 将颜色通道顺序从RGBA转换为BGR，因为PIL颜色通道顺序是RGBA，而cv2颜色通道顺序是BGR
        out.write(new_img_array)

    def step5(self,weight_path,vsr_input_video_path,vsr_output_video_path):
        vsr.super_resolve_video(weight_path,vsr_input_video_path,vsr_output_video_path)

#



