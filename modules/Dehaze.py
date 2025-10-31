import cv2
import numpy as np
import os


#计算雾化图像的暗通道
def dark_channel(img, size = 15):
    r, g, b, a = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))#取最暗通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img,kernel)
    return dc_img

#估计全局大气光值
# def get_atmo(img, percent = 0.001):
#     mean_perpix = np.mean(img, axis = 2).reshape(-1)
#     mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
#     return np.mean(mean_topper)


def get_atmo(img, percent=0.001):
    dark = dark_channel(img)  # 计算暗通道
    flat_dark = dark.flatten()
    top_pixels = flat_dark[np.argsort(flat_dark)[-int(len(flat_dark) * percent):]]
    A = np.mean(top_pixels)  # 使用靠近最大值的像素来估计大气光
    return A

#估算透射率图
def get_trans(img, atom, w = 0.4):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t

#引导滤波
def guided_filter(p, i, r, e):
    """
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    """
    #1
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    #2
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    #3
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    #4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    #5
    q = mean_a * i + mean_b
    return q




def dehaze(im):

    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
    atom = get_atmo(img)
    trans = get_trans(img, atom)
    trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
    trans_guided = cv2.max(trans_guided, 0.25)
    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
    return result*255

def dehaze_V2(originPath,savePath):
    '''originaPath:文件夹的路径，图片上一级
       savePath：同理'''
    """for image_name in os.listdir(originPath):
        image_path = os.path.join(originPath,image_name)
        print(image_path)
        im = cv2.imread(image_path)
        img = im.astype('float64') / 255
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
        atom = get_atmo(img)
        trans = get_trans(img, atom)
        trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
        trans_guided = cv2.max(trans_guided, 0.25)
        result = np.empty_like(img)
        for i in range(3):
            result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
        oneSave = os.path.join(savePath,image_name)
        cv2.imwrite(oneSave, result*255)"""
    
    for image_name in os.listdir(originPath):
        image_path = os.path.join(originPath, image_name)
        print(image_path)
        im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 读取带有透明通道的图像
        #cv2.imwrite(r'C:/Users/Violet/Desktop/facial/change_chaofen/1.png', im) 
        img = im.astype('float64') / 255  # 只处理RGB通道
        #cv2.imwrite(r'C:/Users/Violet/Desktop/facial/change_chaofen/2.png', im[:, :, :3]) 
        alpha_channel = im[:, :, 3]  # 获取alpha通道
        alpha_channel[alpha_channel != 0] = 255
        #a = np.dstack((img * 255, alpha_channel)).astype(np.uint8)
        #cv2.imwrite(r'C:/Users/Violet/Desktop/facial/change_chaofen/3.png', a) 

        img_gray = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY).astype('float64') / 255
        atom = get_atmo(img)
        trans = get_trans(img, atom)
        trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
        trans_guided = cv2.max(trans_guided, 0.25)
        result = np.empty_like(img)
        for i in range(4):
            result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom

        # 合并去雾结果和原始的alpha通道
        result_with_alpha = np.dstack((result[:,:,:3] * 255, alpha_channel)).astype(np.uint8)
        
        oneSave = os.path.join(savePath, image_name)
        cv2.imwrite(oneSave, result*255)  # 保存结果，保留透明度 
        
        
    # cv2.imshow("source",img)
    # cv2.imshow("result", result)
    
    #cv2.waitKey(0)       


#dehaze_V2('input/set1','output')

# transform_mode=['b','a','e','i','o','u']
# transform_strength=[1.0,0.75,0.5,0.25]
"""transform_mode=['b','a','i','o']
transform_strength=[1.0,0.5]


for mode in transform_mode:
    print(f"###################################################开始去朦胧{mode}表情###################################################")
    if mode=='b':
        strength=0.0
        print(f"###################################################开始去朦胧{mode}表情的强度为{strength}###################################################")
        if not os.path.exists(f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}'):
            os.makedirs(f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}')
        image_path = f'C:/Users/Violet/Desktop/facial/beishang2/{mode}_{strength}'
        #image_path = f'C:/Users/Violet/Desktop/facial/change_chaofen/{mode}_{strength}'
        print(image_path)
        output_path = f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}'
        dehaze_V2(image_path,output_path)

    else:

        for strength in transform_strength:
            if  mode=='i' or mode=='o':
             strength=1.0
            print(f"###################################################开始去朦胧{mode}表情的强度为{strength}###################################################")
            if not os.path.exists(f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}'):
                os.makedirs(f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}')
            image_path = f'C:/Users/Violet/Desktop/facial/beishang2/{mode}_{strength}'
            print(image_path)
            output_path = f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}'
            dehaze_V2(image_path,output_path)"""
# if not os.path.exists(f'C:/Users/Violet/Desktop/facial/source_picture-3'):
#         os.makedirs(f'C:/Users/Violet/Desktop/facial/source_picture-3')
# image_path = f'C:/Users/Violet/Desktop/facial/source_picture-2'
# print(image_path)
# output_path = f'C:/Users/Violet/Desktop/facial/source_picture-3'
# dehaze_V2(image_path,output_path)
