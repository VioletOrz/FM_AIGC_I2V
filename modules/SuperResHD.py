import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('.')+'/lib')
sys.path.append(os.path.abspath('.')+'/lib/Real_ESRGAN')
from lib.Real_ESRGAN.realesrgan import RealESRGANer
#from lib.Real_ESRGAN.realesrgan.archs.srvgg_arch import SRVGGNetCompact
from PIL import Image
import os
import cv2
from tqdm import tqdm


def main(dir_path,output_path,):
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=dir_path, help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus_anime_6B',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default=output_path, help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')

    current_working_dir = os.getcwd()

    parser.add_argument(
        '--model_path', type=str, default=current_working_dir+"/lib/Real_ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth", help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # determine model paths

    model_path = os.path.join(current_working_dir + '/lib/Real_ESRGAN/weights', args.model_name + '.pth')
    print(model_path)
    if not os.path.isfile(model_path):
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(current_working_dir, 'lib/Real_ESRGAN/weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            """if force_extension==None:
                if args.ext == 'auto':
                    extension = extension[1:]
                else:
                    extension = args.ext
                if img_mode == 'RGBA':  # RGBA images should be saved in png format
                    extension = 'png'
                if args.suffix == '':
                    save_path = os.path.join(args.output, f'{imgname}.{extension}')
                else:
                    #save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
                    save_path = os.path.join(args.output, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(args.output, f'{imgname}.{force_extension}')"""
            
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'webp' #'png'
            if args.suffix == '':
                save_path = os.path.join(args.output, f'{imgname}.{extension}')
            else:
                #save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
                save_path = os.path.join(args.output, f'{imgname}.{extension}')
            cv2.imwrite(save_path, output)


def super_resolution(input_file, output_file,force_extension=None):
# if __name__ == '__main__':
#     main('inputs','results')
    transform_mode=['b','a','i','o']
    #transform_mode=['a','e','i','o','u']
    transform_strength=[1.0,0.5]

    for mode in transform_mode:
        print(f"###################################################开始合成{mode}表情###################################################")
        if mode=='b':
            strength=0.0
            # if not os.path.exists(f'C:/Users/Violet/Desktop/facial/change_chaofen/{mode}_{strength}'):
            #     os.makedirs(f'C:/Users/Violet/Desktop/facial/change_chaofen/{mode}_{strength}')
            #image_path = f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}'
            #image_path = f'C:/Users/Violet/Desktop/facial/beishang2/{mode}_{strength}'
            #image_path = f'C:/Users/Violet/Desktop/facial/change_size/{mode}_{strength}'
            image_path = input_file+f'/{mode}_{strength}'
            print(image_path)
            output_path = output_file+f'/{mode}_{strength}'
            main(image_path,output_path,)

        else:
            if mode=='a':
                for strength in transform_strength:
                    # if not os.path.exists(f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}'):
                    #     os.makedirs(f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}')
                    image_path = input_file+f'/{mode}_{strength}'
                    #image_path = f'C:/Users/Violet/Desktop/facial/change_size/{mode}_{strength}'
                    print(image_path)
                    output_path = output_file+f'/{mode}_{strength}'
                    main(image_path,output_path,)

            else:
                strength=1.0
                image_path = input_file+f'/{mode}_{strength}'
                #image_path = f'C:/Users/Violet/Desktop/facial/change_size/{mode}_{strength}'
                print(image_path)
                output_path = output_file+f'/{mode}_{strength}'
                main(image_path,output_path,)




def convert_to_png(image_path):
    # 打开图片
    image = Image.open(image_path)

    # 获取图片的文件名和扩展名
    file_name, file_extension = os.path.splitext(image_path)

    # 如果文件已经是PNG格式，则无需转换
    if file_extension.lower() == '.webp':
        print("图片已经是webp格式")
        return

    # 将图片转换为PNG格式并保存
    new_file_path = file_name + '.webp'
    image.save(new_file_path, 'webp')

    # 删除原文件
    os.remove(image_path)

    #print(f"已将 {image_path} 转换为 {new_file_path}")


# transform_mode=['a','i','o']
# transform_strength=[1.0,0.5]
# # transform_mode=['a']
# # transform_strength=[1]

# for mode in transform_mode:
#         print(f"###################################################开始合成{mode}表情###################################################")


#         for strength in transform_strength:
#             if  mode=='i' or mode=='o':
#                 strength=1.0

#             # images_dir1 = r"C:\Users\FM\Desktop\facial\change_chaofen/b_0.0"
#             # files1 = os.listdir(images_dir1)
#             for i in tqdm(range(100), desc="转webp"):
#                 image_path = f'C:/Users/Violet/Desktop/facial/change_chaofen/{mode}_{strength}/img_{i + 1}.png'
#                 convert_to_png(image_path)
#                 print(image_path)

#dehazed_pack_name: "dehazed_image_package" #Dehazed images output file
#SuperResHD_pack_name: 'SuperResHD_image_package' #Super resolution HD inpaint algorithm

#super_resolution('C:/Users/Violet/Desktop/facial/OOI/dehazed_image_package', 'C:/Users/Violet/Desktop/facial/OOI/SuperResHD_image_package')
'''transform_mode=['b','a','i','o']
#transform_mode=['a','e','i','o','u']
transform_strength=[1.0,0.5]


for mode in transform_mode:
    print(f"###################################################开始合成{mode}表情###################################################")
    if mode=='b':
        strength=0.0
        # if not os.path.exists(f'C:/Users/Violet/Desktop/facial/change_chaofen/{mode}_{strength}'):
        #     os.makedirs(f'C:/Users/Violet/Desktop/facial/change_chaofen/{mode}_{strength}')
        #image_path = f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}'
        #image_path = f'C:/Users/Violet/Desktop/facial/beishang2/{mode}_{strength}'
        #image_path = f'C:/Users/Violet/Desktop/facial/change_size/{mode}_{strength}'
        image_path = f'C:/Users/Violet/Desktop/facial/OOI/dehazed_image_package/{mode}_{strength}'
        print(image_path)
        output_path = f'C:/Users/Violet/Desktop/facial/OOI/SuperResHD_image_package/{mode}_{strength}'
        main(image_path,output_path)

    else:
        if mode=='a':
         for strength in transform_strength:
            # if not os.path.exists(f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}'):
            #     os.makedirs(f'C:/Users/Violet/Desktop/facial/change_menglong/{mode}_{strength}')
            image_path = f'C:/Users/Violet/Desktop/facial/OOI/dehazed_image_package/{mode}_{strength}'
            #image_path = f'C:/Users/Violet/Desktop/facial/change_size/{mode}_{strength}'
            print(image_path)
            output_path = f'C:/Users/Violet/Desktop/facial/OOI/SuperResHD_image_package/{mode}_{strength}'
            main(image_path,output_path)

        else:
            strength=1.0
            image_path = f'C:/Users/Violet/Desktop/facial/OOI/dehazed_image_package/{mode}_{strength}'
            #image_path = f'C:/Users/Violet/Desktop/facial/change_size/{mode}_{strength}'
            print(image_path)
            output_path = f'C:/Users/Violet/Desktop/facial/OOI/SuperResHD_image_package/{mode}_{strength}'
            main(image_path,output_path)'''