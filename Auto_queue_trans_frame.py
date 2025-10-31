import subprocess
import os, sys

# 让 Python 输出使用 UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

python_exe = r"E:/AIGC/env/FM_01/python.exe"

# "100001-1-1","100001-3-1","100001-4-1","100001-5-1","100001-6-1","100001-7-1","100001-8-1","100001-9-1",
#              "100002-1-1","100002-3-1","100002-4-1","100002-5-1","100002-6-1","100002-7-1","100002-8-1","100002-9-1",
#              "100003-1-1","100003-3-1","100003-4-1","100003-5-1","100003-6-1","100003-7-1","100003-8-1","100003-9-1",
#              "100004-1-1","100004-3-1","100004-4-1","100004-5-1","100004-6-1","100004-7-1","100004-8-1","100004-9-1",
#              "100006-1-1","100006-3-1","100006-4-1","100006-5-1","100006-6-1","100006-7-1","100006-8-1","100006-9-1",
start = 0
end = 0
for todo in ["100001-1-1"]:
    print(f"开始处理视频 {todo} 的转帧")

    cmd = [
        python_exe,
        "tools/video_to_trans_frame.py",
        "--video_path",f"./data/ani-pic/{todo}/oragin.mp4",
        "--output_dir", f"./data/trans_frames/{todo}",
        "--start", f"{start}",
        "--end", f"{end}",
        "--fps", "24"  
    ]

    # 执行命令
    result = subprocess.run(cmd, text=True,check=True)

    # 输出执行结果  
    print("=== STDOUT ===")
    print(result.stdout)
    print("=== STDERR ===")
    print(result.stderr)

    # 判断是否执行成功
    if result.returncode == 0:
        print("✅ 执行成功！")
    else:
        print(f"❌ 执行失败，退出码：{result.returncode}")
        

    # 构造命令参数
    cmd = [
        python_exe,
        "launch.py",
        "--config", "./config/sitting.yaml",
        "--pipeline", "IFDSRFc",
        "--is_trans", "True",
        "--use_api", "True",
        "--num_frames", "2",
        "--use_trans_frame_list", "True",
        "--frame_list_dir", f"./data/trans_frames/{todo}/trans",
        "--package_name", f"./fix/{todo}",
        "--input_path", f"./data/trans_frames/{todo}/frame/frame_000.png",
        "--alternate_background", "False",
        "--emotion_from_tensor", "False",
        "--emotion_pose_load_path", "./data/pose/04_01.mp4"
    ]


    # 打印命令（可选）
    print("🚀 正在执行命令：")
    print(" ".join(cmd))

    # 执行命令（会实时输出日志）
    subprocess.run(cmd, check=True)