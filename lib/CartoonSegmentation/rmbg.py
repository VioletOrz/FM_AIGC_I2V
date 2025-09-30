import os
import requests
import numpy as np
from PIL import Image

def get_mask_and_bbox(png_path):
    """
    计算 PNG 图像的 mask 和 bounding box (xywh)
    mask: numpy.ndarray，形状 (H, W)，值为 alpha 通道 (0~255)
    bbox: (x, y, w, h)，框出非完全透明区域
    """
    # 打开图像并确保是 RGBA
    img = Image.open(png_path).convert("RGBA")
    rgba = np.array(img)

    # 取 alpha 通道作为 mask
    mask = rgba[:, :, 3]

    # 找出非透明的像素点
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        # 图像完全透明
        return mask, (0, 0, 0, 0)

    # 计算边界框
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w = x_max - x_min + 1
    h = y_max - y_min + 1
    bbox = (int(x_min), int(y_min), int(w), int(h))

    return mask, bbox

# # 示例使用
# mask, bbox = get_mask_and_bbox("input.png")
# print("mask shape:", mask.shape)
# print("bbox (xywh):", bbox)

def remove_background(image_path, api_key, output_path="photoroom-result.png"):
    """
    使用 PhotoRoom API 移除图像背景。

    参数：
        image_path: 要处理的图像的绝对路径。
        api_key: 你的 PhotoRoom API 密钥。
        output_path: 保存结果图像的路径 (默认为 "photoroom-result.png")。

    返回值：
        如果成功，返回 True；否则返回 False。
    """
    try:
        # 确保文件存在.
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return False

        url = "https://sdk.photoroom.com/v1/segment"
        headers = {"x-api-key": api_key}
        files = {"image_file": open(image_path, "rb")}  # 以二进制读取模式打开文件

        # 发送 POST 请求，包含文件和请求头
        response = requests.post(url, headers=headers, files=files)

        # 检查请求是否成功 (状态码 200 表示成功)
        if response.status_code == 200:
            # 将响应内容 (处理后的图像) 写入文件
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Background removed successfully. Image saved to {output_path}")
            return True
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            print(response.text)  # 打印详细的错误信息
            return False

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False