import subprocess
import os, sys

# 让 Python 输出使用 UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

python_exe = r"E:/AIGC/env/FM_01/python.exe"


for i in range(15, 16):
    todo = f'{i:03d}'
    print(f"开始处理视频 {todo} 的转帧")

    cmd = [
        python_exe,
        "launch.py",
        "--config", "./config/sitting.yaml",
        "--pipeline", "Ic",
        "--is_trans", "True",
        "--use_api", "True",
        "--use_trans_frame_list", "False",
        "--frame_list_dir", f"./data/trans_frames/{todo}/trans",
        "--package_name", f"syq_{todo}",
        "--input_path", f"./data/ani-pic/syq/{todo}.png",
        "--alternate_background", "False",
        "--emotion_from_tensor", "True",
        "--emotion_pose_load_path", "./data/pose/x1_01.mp4"
    ]

    # 打印命令（可选）
    print("🚀 正在执行命令：")
    print(" ".join(cmd))

    # 执行命令（会实时输出日志）
    subprocess.run(cmd, check=True)

# cmd = [
#     python_exe,
#     "launch.py",
#     "--config", "./config/sitting.yaml",
#     "--pipeline", "IFDSRFc",
#     "--is_trans", "True",
#     "--use_api", "True",
#     "--use_trans_frame_list", "False",
#     "--frame_list_dir", f"./data/trans_frames/{todo}/trans",
#     "--package_name", f"{todo}_l",
#     "--input_path", f"./data/ani-pic/{todo}/oragin.png",
#     "--alternate_background", "False",
#     "--emotion_from_tensor", "True",
#     "--emotion_pose_load_path", "./data/pose/04_01.mp4"
# ]

# # 打印命令（可选）
# print("🚀 正在执行命令：")
# print(" ".join(cmd))

# 执行命令（会实时输出日志）
# subprocess.run(cmd, check=True)