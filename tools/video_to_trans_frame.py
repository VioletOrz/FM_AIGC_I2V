import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../lib/CartoonSegmentation"))
)
from lib.CartoonSegmentation.rmbg_video import extract_frames_and_cut, duplicate_images_reversed
import argparse
def main():
    # ===== 解析命令行参数 =====
    parser = argparse.ArgumentParser(description="视频抠图+倒序帧扩展工具")
    parser.add_argument("--video_path", type=str, required=True, help="输入视频路径")
    # parser.add_argument("--api_key", type=str, required=True, help="PhotoRoom API key")
    parser.add_argument("--output_dir", type=str, required=True, help="输出文件夹路径")
    parser.add_argument("--fps", type=int, default=16, help="抽帧帧率（默认16）")
    parser.add_argument("--start", type=int, default=0, help="倒序插入起始帧编号（默认0）")
    parser.add_argument("--end", type=int, default=99, help="倒序插入结束帧编号（默认99）")
    args = parser.parse_args()
    api_key = 'sk_pr_default_b56420e7854b3d179082e4e9874f20bb30dfaecd' #Your remove.bg API key

    # ===== 调用主流程 =====
    print("开始抽帧并抠图 …")
    extract_frames_and_cut(
        args.video_path,
        api_key,
        args.output_dir,
        fps_extract=args.fps,
        start=args.start,
        end = args.end
    )

    # print("开始生成倒序帧 …")
    # duplicate_images_reversed(
    #     os.path.join(args.output_dir, "trans"),
    #     prefix="img_",
    #     start=args.start,
    #     end=args.end,
    #     ext=".png"
    # )

    print("全部完成！输出目录:", args.output_dir)


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     video_path = "./data/ani-pic/348/oragin.mp4"
#     api_key = 'e9ff7d61ffa318d0c6eb59498c4a6dd36b143898' #Your remove.bg API key
#     output_dir = "./data/trans_frames/348"
#     fps = 16
    
#     extract_frames_and_cut(video_path, api_key, output_dir, fps_extract=fps, expand_ratio=0.2)

#     duplicate_images_reversed(os.path.join(output_dir, "trans"), prefix="img_", start=0, end=49, ext=".png")
