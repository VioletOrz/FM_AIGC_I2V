##############################################################################################
#这个代码用来分割长视频的，可以把视频分割为30fps 4秒一段 将长素材分割成短的 以提供更多样的motion动作
##############################################################################################

import os
import cv2

def split_video(input_video_path, output_folder, target_fps=30, segment_frames=120):
    # 获取视频文件名（不带扩展名）
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    
    # 获取原视频的帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 设置视频读取为 30 fps
    frame_interval = int(original_fps / target_fps)

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建保存视频的文件夹
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    # 变量初始化
    segment_count = 0
    remaining_frames = total_frames
    current_frame = 0
    
    while current_frame < total_frames:
        # 计算当前段的保存路径
        segment_count += 1
        if segment_count <= 10:
            output_path = os.path.join(video_output_folder, f"{video_name}_{segment_count:02d}.mp4")
        else:
            output_path = os.path.join(video_output_folder, f"{video_name}_res.mp4")
        
        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (int(cap.get(3)), int(cap.get(4))))

        # 每个段的帧数为 segment_frames
        for _ in range(segment_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame % frame_interval == 0:
                out.write(frame)
            current_frame += 1
        
        out.release()

        remaining_frames -= segment_frames
        if remaining_frames <= 0:
            break
    
    # 释放视频对象
    cap.release()

# 使用示例
input_video_path = r"D:\FM_AIGC_I2V\data\emotion\x1.mp4"
output_folder = r"D:\FM_AIGC_I2V\data\emotion"
split_video(input_video_path, output_folder)