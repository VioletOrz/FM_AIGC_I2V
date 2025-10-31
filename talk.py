import os
import cv2
from PIL import Image

def make_mouth_video_by_index(base_dir, mouth_sequence, output_path="output_index.mp4", fps=12):
    """
    根据口型序列与帧索引生成视频。
    第 i 帧取 mouth_sequence[i] 文件夹中的第 i 张图片。

    参数:
        base_dir (str): 包含所有口型文件夹的根目录
        mouth_sequence (list[str]): 口型序列，如 ["a_0.5", "a_1.0", "b_0.0", ...]
        output_path (str): 输出视频路径
        fps (int): 视频帧率
    """
    # 收集所有口型文件夹路径
    mouth_folders = {name: os.path.join(base_dir, name) for name in set(mouth_sequence)}

    # 检查文件夹是否存在
    for name, path in mouth_folders.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(f"❌ 找不到文件夹: {path}")

    # 预读取各口型下的文件列表并排序
    mouth_images = {
        name: sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(('.webp', '.png', '.jpg', '.jpeg'))
        ])
        for name, path in mouth_folders.items()
    }

    # 获取第一张图确定分辨率
    first_img_path = mouth_images[mouth_sequence[0]][0]
    first_img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"🎬 开始生成视频：{output_path}")
    print(f"帧率：{fps}，分辨率：{w}x{h}")

    for i, mouth in enumerate(mouth_sequence):
        img_list = mouth_images[mouth]
        if i >= len(img_list):
            print(f"⚠️ {mouth} 中帧数不足，使用最后一帧代替（索引 {i}）")
            frame_path = img_list[-1]
        else:
            frame_path = img_list[i]

        frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"⚠️ 无法读取帧: {frame_path}")
            continue

        # 转换透明图像为RGB
        if frame.shape[2] == 4:
            alpha = frame[:, :, 3] / 255.0
            bg = (255 * (1 - alpha)).astype(frame.dtype)
            frame = frame[:, :, :3] * alpha[:, :, None] + bg[:, :, None]
            frame = frame.astype('uint8')

        frame = cv2.resize(frame, (w, h))
        video_writer.write(frame)

    video_writer.release()
    print(f"✅ 视频生成完成，共 {len(mouth_sequence)} 帧。输出：{output_path}")

if __name__ == "__main__":
    base_dir = r"package\418\video_01\resized_image_package"
    mouth_sequence = [
        "a_0.5", "a_1.0", "i_1.0", "b_0.0", "a_1.0",
        "o_1.0", "a_0.5", "b_0.0", "i_1.0", "a_1.0",
        "a_0.5", "b_0.0", "o_1.0", "a_1.0", "i_1.0",
        "b_0.0", "a_0.5", "a_1.0", "i_1.0", "o_1.0",
        "b_0.0", "a_1.0", "a_0.5", "i_1.0", "b_0.0",
        "a_1.0", "o_1.0", "b_0.0", "i_1.0", "a_0.5",
        "a_1.0", "b_0.0", "o_1.0", "i_1.0", "a_1.0",
        "b_0.0", "a_0.5", "i_1.0", "o_1.0", "a_1.0",
        "a_1.0", "b_0.0", "o_1.0", "i_1.0", "a_1.0",
        "b_0.0", "a_0.5", "i_1.0", "o_1.0", "a_1.0",
        
    ]
    # 每个口型持续两帧（重复两次）
    mouth_sequence_2x = [m for m in mouth_sequence for _ in range(8)]
    output_path = "output1.mp4"
    fps = 24

    make_mouth_video_by_index(base_dir, mouth_sequence, output_path, fps)
