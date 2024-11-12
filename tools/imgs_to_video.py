################################################################################
#这个脚本可以将图包中的b图包（不带口型）合成为视频
################################################################################

import cv2
import os
import re

def create_video_from_images(folder_path, output_video_path, fps=30):
    # 获取指定目录下符合 img_001 ~ img_999 格式的文件名并排序
    image_files = sorted([f for f in os.listdir(folder_path) 
                          if re.match(r'img_\d{3}\.webp$', f)], 
                         key=lambda x: int(x[4:7]))

    # 如果没有符合条件的图片，退出函数
    if not image_files:
        print("No images found in the specified format.")
        return

    # 读取第一张图片以获取尺寸信息
    first_image_path = os.path.join(folder_path, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, channels = frame.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4格式
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐帧写入图片到视频
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 释放视频写入对象
    video_writer.release()
    print(f"Video saved at {output_video_path} with FPS: {fps}")

def images_to_video(image_folder, output_video_path, fps=30):
    # 获取符合命名格式的图片文件并按文件名排序
    images = [img for img in os.listdir(image_folder) if img.startswith("img_") and img.endswith(".webp")]
    images.sort()

    # 检查图片是否存在
    if not images:
        print("没有符合条件的图片")
        return

    # 读取第一张图片以获取视频的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # 定义视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐帧将图片写入视频
    for image_file in images:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图片 {image_file}")
            continue
        video.write(frame)
    
    # 释放视频对象
    video.release()
    print(f"视频已保存到 {output_video_path}")

# 使用指定的文件夹路径、输出视频路径和帧率
if __name__ == "__main__":
    folder_path = r"C:\Users\Violet\Desktop\facial\test_trans\base_image_package\a_1.0"
    output_video_path = r"C:\Users\Violet\Desktop\facial/output_video.mp4"
    fps = 30  # 自定义帧率
    images_to_video(folder_path, output_video_path, fps=30)
    #create_video_from_images(folder_path, output_video_path, fps)
