#这里面在原有的基础上增加了人脸识别以及裁剪的模块，人脸识别主函数可以看detect.py这一个文件

from PIL import Image
import os
import numpy as np
import sys

from animeinsseg import AnimeInsSeg  # 假设 AnimeInsSeg 是您的检测器
import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from rmbg import remove_background, get_mask_and_bbox

z=0

API_KEY = "e9ff7d61ffa318d0c6eb59498c4a6dd36b143898"
def change_position(img_path,cropped_dir=''):
  
  def detect(save_img=False, cropped_dir = ''):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    global z

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    print(weights)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        c1,c2,length=plot_one_box(z,xyxy, im0, color=colors[int(cls)], line_thickness=3, save_path=cropped_dir)
                        z=z+1
    return c1,c2,length

  
  parser = argparse.ArgumentParser()
  current_working_dir = os.getcwd()
  parser.add_argument('--weights', nargs='+', type=str, default=current_working_dir+'/lib/CartoonSegmentation/yolov5x_anime.pt', help='model.pt path(s)')
  parser.add_argument('--source', type=str, default=img_path, help='source')  # file/folder, 0 for webcam
  parser.add_argument('--output', type=str, default=current_working_dir+'/lib/CartoonSegmentation/output', help='output folder')  # output folder
  parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
  parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
  parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
  parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
  parser.add_argument('--view-img', action='store_true', help='display results')
  parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
  parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
  parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
  parser.add_argument('--augment', action='store_true', help='augmented inference')
  parser.add_argument('--update', action='store_true', help='update all models')
  #parser.add_argument('--config', default="./config/sitting.yaml", type=str, required=False, help='Path to the config file.')
  #parser.add_argument('--pipeline', type=str, default="PIFDFSFRF",required=False, help='Pipeline.')
  #parser.add_argument('--is_trans', type=str, default="None", required=False, help='Generate a transparent no background image package.')
  #parser.add_argument('--package_name', type=str, default=None, required=False, help='Output package name.')
  opt, unknown_args = parser.parse_known_args()
  print(opt)

  with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            c1,c2,length=detect(cropped_dir=cropped_dir)
  
  return c1,c2,length

def remove_masked_area(img, mask):

    # 将 img 转为 numpy 数组
    img = img.convert("RGBA")
    img_array = np.array(img)
    
    # 将 mask 应用于 img，扣掉被 mask 覆盖的区域，设置为透明
    img_array[mask, 3] = 0  # 将 Alpha 通道（透明度）设为 0 表示完全透明

    # 转回 PIL Image
    result_img = Image.fromarray(img_array)

    return result_img
def remove_masked_areafan(img, mask):

    # 将 img 转为 numpy 数组
    mask = ~mask  # 反转 mask
    img = img.convert("RGBA")
    img_array = np.array(img)
    
    # 将 mask 应用于 img，扣掉被 mask 覆盖的区域，设置为透明
    img_array[mask, 3] = 0  # 将 Alpha 通道（透明度）设为 0 表示完全透明

    # 转回 PIL Image
    result_img = Image.fromarray(img_array)

    return result_img

def read_json_file(json_path:str):
    """
    读取json文件并返回数据
    json_path: json文件的路径
    """
    import json

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 解析 JSON 文件内容为 Python 数据结构
    return data

def write_json_file(json_path:str, data):
    """
    将数据写入json文件
    json_path: json文件的路径
    data: 需要写入的数据
    """
    import json

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_masks(img_path, c1, c2, output_dir, rmbg_image_path, cropped_dir='', API_KEY=None, idx = 0, bdbox = None):
    
    img = Image.open(img_path).convert('RGBA')
    # use_api = True
    # API_KEY = 'e9ff7d61ffa318d0c6eb59498c4a6dd36b143898'


    # remove_background(image_path=img_path, api_key=API_KEY, output_path=cropped_dir + "photoroom-result.png")
    mask_api, _ = get_mask_and_bbox(rmbg_image_path)

        # instances.masks = [mask.astype(bool)]  # 将 mask 转换为布尔类型并存储在列表中
    # 如果实例不为空
    if True:

        # mask = instances.masks[0]
        mask_img = Image.fromarray(mask_api.astype(np.uint8))

        # 创建一个新的透明图像
        transparent_img = Image.new('RGBA', img.size)

        # 使用掩码将原始图像的对应部分抠出来，并计算掩码的边界框
        left, top, right, bottom = img.width, img.height, 0, 0
        for x in range(img.width):
            for y in range(img.height):
                if mask_img.getpixel((x, y)) > 0:
                    transparent_img.putpixel((x, y), img.getpixel((x, y)))
                    left, top, right, bottom = min(left, x), min(top, y), max(right, x), max(bottom, y)#left和top是在遮罩左上角在原图中的坐标
        if bdbox:
            left = min(left, bdbox[0])
            top = min(top, bdbox[1])
            right = max(right, bdbox[2])
            bottom = max(bottom, bdbox[3])
        # result = remove_masked_area(img, mask)
        # result.save(cropped_dir + 'output1.png')
        # result1 = remove_masked_areafan(img,mask)
        # result1.save(cropped_dir + 'output2.png')
        # 裁剪图像
        cropped_img = transparent_img.crop((left, top, right, bottom))
        
        # 创建一个新的透明图像，大小为边界框的最大边长
        max_side = int(1.2*max(right - left, bottom - top))   #人脸识别移动位置的mask   正常为1.2  可以根据不同情况进行更改
        print(max_side)
        square_img = Image.new('RGBA', (max_side, max_side))
        # change_position1=change_position()
        cropped_img1 = f"cropped_image.png"
        cropped_img.save(os.path.join(cropped_dir, cropped_img1))
        image_path=cropped_dir+"cropped_image.png"
        # c1,c2,length=change_position(image_path, cropped_dir)
        #新图像脸的中心位置
        x=int(max_side//2)
        y=int(max_side//4)
        x_relative=int(((max_side - cropped_img.width) // 2)+c1-x)
        y_relative=int(((max_side - cropped_img.height) // 2)+c2-y)
        a=((max_side - cropped_img.width) // 2)-x_relative
        b=((max_side - cropped_img.height) // 2)-y_relative
        # 将裁剪出的图像粘贴到新图像的中心
        square_img.paste(cropped_img, (a, b))#给出的是square_img的被粘贴位置的左上角坐标，其中一定有一个是0，因为max_side就是width和height的最大值

        # 为掩码生成一个保存名
        mask_name = f"mask_{idx:03d}.png"
        img_name = f"img_{idx:03d}.png"

        # 确保输出目录存在
        if not os.path.exists(os.path.join(output_dir,'trans')):
            os.makedirs(os.path.join(output_dir,'trans'))

        # 将掩码和抠出的图像保存到 output_dir 目录下
        mask_img.save(os.path.join(os.path.join(output_dir,'frame'), mask_name))
        square_img.save(os.path.join(os.path.join(output_dir,'trans'), img_name))
        #os.remove('output_dir/cropped_image.png')
        global z          
        z = 0
        return left, top, right, bottom#返回蒙版左上角坐标，以及拼到正方形后左上角坐标，以及蒙版的长和宽


# 使用函数
# save_masks(r"D:\FaceMind_AIGC\CartoonSegmentation\00136-1735163985.png", "output_dir")


def crop_image(image_path,cropped_dir):
    c1,c2,length=change_position(image_path,cropped_dir)
    # 打开图像
    image = Image.open(image_path)
    x1=c1-3*length
    y1=c2-2*length
    x2=c1+3*length
    y2=c2+4*length
    # print(x1)
    # print(y1)
    # print(x2)
    # print(y2)
    if x1<0:
        x2=x2-x1
        x1=0
    if y1<0:
        y2=y2-y1
        y1=0
    if x2>image.width:
        x1=x1-x2+image.width
        x2=image.width
        if x1<0:
            y2=y2+x1
            x1=0
    if y2>image.height:
        y1=y1-y2+image.height
        y2=image.height

    # print(x1)
    # print(y1)
    # print(x2)
    # print(y2)
    

    # 裁剪图像
    cropped_image=image.crop((x1, y1, x2, y2))
    #cropped_image=image[y1:y2, x1:x2]
    img_path= cropped_dir + "/cropped_image_half.png"
    # 保存裁剪后的图像
    cropped_image.save(img_path)

    return img_path


def extract_frames_and_cut(video_path, api_key, output_dir="output_frames",
                           fps_extract=2, start=0, end=49):
    os.makedirs(os.path.join(output_dir,'frame'), exist_ok=True)
    os.makedirs(os.path.join(output_dir,'trans'), exist_ok=True)
    os.makedirs('./data/cropped_dir_trans', exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps_extract <= 0:
        fps_extract = 1
    frame_interval = max(int(round(fps / fps_extract)), 1)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞 视频帧率: {fps:.2f}, 共 {frame_count} 帧, 每 {frame_interval} 帧抽一张")

    frame_idx = 0
    idx = 0
    save_idx = 0

    left, top, right, bottom = None, None, None, None
    bdbox_list = {}
    if start != 0:
        bdbox_list = read_json_file(os.path.join(output_dir, 'bdbox_list.json'))
        left, top, right, bottom = bdbox_list[str(start - 1)]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        
        if frame_idx % frame_interval == 0:
            temp_jpg = os.path.join(output_dir, f"frame/frame_{idx:03d}.png")
            temp_png = os.path.join(output_dir, f"frame/trans_{idx:03d}.png")
            cv2.imwrite(temp_jpg, frame)

            if idx >= start:
                ok = remove_background(temp_jpg, api_key, temp_png)
            else:
                ok = False

            if save_idx == 0 and ok:
            #     # 初始化检测器
            #     current_working_dir = os.getcwd()
            #     detector = AnimeInsSeg(current_working_dir+"/lib/CartoonSegmentation/models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt",device='cuda')
            #     detector.init_tagger()

            #     # 使用检测器获取实例
            #     instances = detector.infer(temp_jpg, output_type='numpy', infer_tags=False, pred_score_thr=0.1)
                c1,c2,length=change_position(os.path.join(output_dir, f"frame/trans_000.png"), './data/cropped_dir_trans/')
                print(c1,c2)
            
            if ok:
                # 复刻 save_masks 的补空白方式
                # bdbox有bug
                left, top, right, bottom = save_masks(temp_png, c1, c2, output_dir, rmbg_image_path = temp_png, cropped_dir='./data/cropped_dir_trans/', API_KEY=api_key, idx = idx, bdbox = (left, top, right, bottom) if left is not None else None)
                # if start == 0:
                bdbox_list[idx] = (left, top, right, bottom)
                print(f"✅ 已保存: {temp_png}")
                write_json_file(os.path.join(output_dir, 'bdbox_list.json'), bdbox_list)
                save_idx += 1
            else:
                print(f"❌ 抠图失败: {temp_jpg}")
                if idx > start:
                    break
            
            idx += 1
            if idx > end:  # 测试用
                break

        frame_idx += 1
    write_json_file(os.path.join(output_dir, 'bdbox_list.json'), bdbox_list)
    cap.release()
    print(f"✅ 完成！共保存 {idx} 张抠图。")

import os
import shutil

def duplicate_images_reversed(src_dir, prefix="img_", start=0, end=49, ext=".png"):
    """
    将 img_start ~ img_end 的图像倒序复制并编号为 end+1 ~ end*2+1
    例如：img_0~img_50 -> 倒序插入 img_51~img_101
    """
    for i, src_idx in enumerate(range(end, start - 1, -1), start=end + 1):
        src_name = f"{prefix}{src_idx:03d}{ext}"
        dst_name = f"{prefix}{i:03d}{ext}"
        src_path = os.path.join(src_dir, src_name)
        dst_path = os.path.join(src_dir, dst_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"✅ 复制 {src_name} → {dst_name}")
        else:
            print(f"⚠️ 跳过: {src_name} 不存在")

    print("🎉 倒序插入完成！")

# ===== 示例用法 =====
# 假设所有图片都在当前目录下

def rename_images_to_three_digits(folder, prefix="img_"):
    """
    将文件夹中所有 img_x.* 文件重命名为三位编号形式，如：
    img_0.png → img_000.png
    img_12.jpg → img_012.jpg
    """
    files = [f for f in os.listdir(folder) if f.startswith(prefix)]
    files.sort(key=lambda x: int(x[len(prefix):].split('.')[0]))  # 按数字排序

    for f in files:
        name, ext = os.path.splitext(f)
        try:
            idx = int(name[len(prefix):])
        except ValueError:
            print(f"⚠️ 跳过非数字文件: {f}")
            continue

        new_name = f"{prefix}{idx:03d}{ext}"
        old_path = os.path.join(folder, f)
        new_path = os.path.join(folder, new_name)

        # 仅在需要时重命名
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"✅ {f} → {new_name}")

    print("🎉 重命名完成！")



if __name__ == "__main__":
    # video_path = r"F:\FM_AIGC_I2V\data\ani-pic\344\1.mp4"
    # api_key = 'e9ff7d61ffa318d0c6eb59498c4a6dd36b143898' #Your remove.bg API key
    # output_dir = "./data/trans_frames"
    
    # extract_frames_and_cut(video_path, api_key, output_dir, fps_extract=16, expand_ratio=0.2)

    # duplicate_images_reversed(r"F:\FM_AIGC_I2V\img", prefix="img_", start=0, end=50, ext=".png")
    # ==== 示例 ====
    rename_images_to_three_digits(r"F:\FM_AIGC_I2V\img", prefix="frame_")