# import cv2
# import torch
# import os
# import sys
# sys.path.append('D:\\FaceMind_AIGC\\APISR')
# from test_code import inference
# from test_code.test_utils import load_grl, load_rrdb, load_dat, load_cunet
# from tqdm import tqdm
# '''
# # 加载超分辨率模型
# weight_path="D:\\FaceMind_AIGC\\APISR\\pretrained\\2x_APISR_RRDB_GAN_generator.pth"
# generator = inference.load_rrdb(weight_path, scale=2)
# generator=generator.to(torch.float32)

# # 打开视频
# cap = cv2.VideoCapture('D:\\FaceMind_AIGC\\talking\\tha3\\app\\video_super_res\\input.mp4')

# # 获取视频的帧率和尺寸
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # 创建一个新的VideoWriter对象
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('D:\\FaceMind_AIGC\\talking\\tha3\\app\\video_super_res\\appoutput.mp4', fourcc, fps, (frame_width, frame_height))

# # 逐帧读取视频
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 将帧保存为临时文件
#     cv2.imwrite('temp.png', frame)

#     # 对帧进行超分辨率处理
#     inference.super_resolve_img(generator, 'temp.png', 'temp.png', weight_dtype=torch.float32, downsample_threshold=-1, crop_for_4x=False)

#     # 读取处理后的帧
#     frame_sr = cv2.imread('temp.png')
#     frame_sr = cv2.resize(frame_sr, (frame_width, frame_height))

#     # 将处理后的帧写入新视频
#     out.write(frame_sr)

# # 释放资源
# cap.release()
# out.release()
# '''

# def super_resolve_video(weight_path, input_video_path, output_video_path):
#     # 加载模型权重
#     if weight_path=="D:\\FaceMind_AIGC\\APISR\\pretrained\\2x_APISR_RRDB_GAN_generator.pth":
#         generator = inference.load_rrdb(weight_path, scale=2)
#     elif weight_path=="D:\\FaceMind_AIGC\\APISR\\pretrained\\4x_APISR_RRDB_GAN_generator.pth":
#         generator=inference.load_rrdb(weight_path, scale=4)
#     else:
#         generator=inference.load_grl(weight_path, scale=4)
#     generator = generator.to(torch.float32)

#     # 打开视频
#     cap = cv2.VideoCapture(input_video_path)

#     # 获取视频的帧率和尺寸
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # 创建一个新的VideoWriter对象
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width*2, frame_height*2))
#     # Get total number of frames in the video for progress bar
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Initialize progress bar
#     progress_bar = tqdm(total=total_frames, desc='Processing Video', unit='frame')

#     # 逐帧读取视频
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         newframe_height=frame_height//2
#         newframe_width=frame_width//2
#         # while(newframe_height*newframe_width>=2560*1440):
#         #     newframe_height=newframe_height//2
#         #     newframe_width=newframe_width//2
#         frame=cv2.resize(frame,(newframe_width,newframe_height))
#         # 将帧保存为临时文件
#         cv2.imwrite('temp.png', frame)
        
#         # 对帧进行超分辨率处理
            
#         inference.super_resolve_img(generator, 'temp.png', 'temp.png', weight_dtype=torch.float32, downsample_threshold=-1, crop_for_4x=False)

#         # 读取处理后的帧
#         frame_sr = cv2.imread('temp.png')
#         # frame_sr = cv2.resize(frame_sr, (frame_width, frame_height))

#         # 将处理后的帧写入新视频
#         out.write(frame_sr)
#         progress_bar.update(1)

#     # 释放资源
#     cap.release()
#     out.release()

# weight_path="D:\\FaceMind_AIGC\\APISR\\pretrained\\2x_APISR_RRDB_GAN_generator.pth"
# input_video_path=r"C:\Users\FM\Desktop\11.mp4"
# output_video_path=r"C:\Users\FM\Desktop\11-output.mp4"
# super_resolve_video(weight_path, input_video_path, output_video_path)


import cv2
import torch
import os
import sys
sys.path.append('D:\\FaceMind_AIGC\\APISR')
from test_code import inference
#from test_code.test_utils import load_grl, load_rrdb, load_dat, load_cunet
'''
# 加载超分辨率模型
weight_path="D:\\FaceMind_AIGC\\APISR\\pretrained\\2x_APISR_RRDB_GAN_generator.pth"
generator = inference.load_rrdb(weight_path, scale=2)
generator=generator.to(torch.float32)

# 打开视频
cap = cv2.VideoCapture('D:\\FaceMind_AIGC\\talking\\tha3\\app\\video_super_res\\input.mp4')

# 获取视频的帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建一个新的VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('D:\\FaceMind_AIGC\\talking\\tha3\\app\\video_super_res\\appoutput.mp4', fourcc, fps, (frame_width, frame_height))

# 逐帧读取视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧保存为临时文件
    cv2.imwrite('temp.png', frame)

    # 对帧进行超分辨率处理
    inference.super_resolve_img(generator, 'temp.png', 'temp.png', weight_dtype=torch.float32, downsample_threshold=-1, crop_for_4x=False)

    # 读取处理后的帧
    frame_sr = cv2.imread('temp.png')
    frame_sr = cv2.resize(frame_sr, (frame_width, frame_height))

    # 将处理后的帧写入新视频
    out.write(frame_sr)

# 释放资源
cap.release()
out.release()
'''

def super_resolve_video(weight_path, input_video_path, output_video_path):
    # 加载模型权重
    if weight_path=="D:\\FaceMind_AIGC\\APISR\\pretrained\\2x_APISR_RRDB_GAN_generator.pth":
        generator = inference.load_rrdb(weight_path, scale=2)
    elif weight_path=="D:\\FaceMind_AIGC\\APISR\\pretrained\\4x_APISR_RRDB_GAN_generator.pth":
        generator=inference.load_rrdb(weight_path, scale=4)
    else:
        generator=inference.load_grl(weight_path, scale=4)
    generator = generator.to(torch.float32)

    # 打开视频
    cap = cv2.VideoCapture(input_video_path)

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建一个新的VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width*2, frame_height*2))

    # 逐帧读取视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        newframe_height=frame_height
        newframe_width=frame_width
        while(newframe_height*newframe_width>=2560*1440):
            newframe_height=newframe_height//2
            newframe_width=newframe_width//2
        frame=cv2.resize(frame,(newframe_width,newframe_height))
        # 将帧保存为临时文件
        cv2.imwrite('temp.png', frame)
        
        # 对帧进行超分辨率处理
            
        inference.super_resolve_img(generator, 'temp.png', 'temp.png', weight_dtype=torch.float32, downsample_threshold=-1, crop_for_4x=False)

        # 读取处理后的帧
        frame_sr = cv2.imread('temp.png')
        # frame_sr = cv2.resize(frame_sr, (frame_width, frame_height))

        # 将处理后的帧写入新视频
        out.write(frame_sr)

    # 释放资源
    cap.release()
    out.release()