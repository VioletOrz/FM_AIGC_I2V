from tools.copy_select_img_pack import copy_specified_folders_with_structure, copy_specified_folders_with_structure_deep
import shutil
import os
from Oss_func import Oss_func

oss = Oss_func()
for todo in ["100001", "100002", "100003", "100004", "100006"]:
    len = 9

    traget_dir = f'./package/{todo}-1'
    trans_img_s_dir = f'./facial/{todo}_s_trans'
    trans_img_l_dir = f'./facial/{todo}_l_trans'
    traget_img_l_dir = f'./package/{todo}-1/image_l'
    traget_img_s_dir = f'./package/{todo}-1/image_s'

    copy_specified_folders_with_structure(trans_img_s_dir, traget_img_s_dir, ['Difference_R', "resized_image_package"])
    copy_specified_folders_with_structure(trans_img_l_dir, traget_img_l_dir, ['Difference_R', "resized_image_package"])
    # copy_specified_folders_with_structure(trans_vido_dir, traget_dir, ['cropped_dir'])

    video_list = []
    target_list = []

    for i in range(1, len+1):
        video_list.append(todo + "-" + str(i) + '-1')

    for i in range(len):
        target_list.append(f'video_{i+1}')

    for vid, tar in zip(video_list, target_list):
        copy_specified_folders_with_structure(f'./facial/{vid}_trans', f'./package/{todo}-1/{tar}', ['Difference_R', "resized_image_package"])
        src = f"./data/ani-pic/{vid}/oragin.mp4"     # 源文件路径
        dst = os.path.join(traget_dir, f'{tar}.mp4')  # 目标文件路径
        try:
            shutil.copy(src, dst)
        except FileNotFoundError:
            pass

    src = f"./data/ani-pic/{todo}-1/oragin.png"     # 源文件路径
    dst = os.path.join(traget_dir, 'image.png')  # 目标文件路径
    shutil.copy(src, dst)
    # oss.up_to_oss(f'bucket-digital-human/cdn_dir/human_platform_bot_roles/{todo}_1', './package/380')



# for todo in ["100001", "100002", "100003", "100004", "100006"]:
#     len = 16

#     traget_dir = f'./package/{todo}'
#     trans_img_s_dir = f'./facial/{todo}_s_trans'
#     trans_img_l_dir = f'./facial/{todo}_l_trans'
#     traget_img_l_dir = f'./package/{todo}/image_l'
#     traget_img_s_dir = f'./package/{todo}/image_s'

#     copy_specified_folders_with_structure(trans_img_s_dir, traget_img_s_dir, ['Difference_R', "resized_image_package"])
#     copy_specified_folders_with_structure(trans_img_l_dir, traget_img_l_dir, ['Difference_R', "resized_image_package"])
#     # copy_specified_folders_with_structure(trans_vido_dir, traget_dir, ['cropped_dir'])

#     video_list = [todo]
#     target_list = []

#     for i in range(1, len):
#         video_list.append(todo + "-" + str(i))

#     for i in range(len):
#         target_list.append(f'video_{i+1}')

#     for vid, tar in zip(video_list, target_list):
#         copy_specified_folders_with_structure(f'./facial/{vid}_trans', f'./package/{todo}/{tar}', ['Difference_R', "resized_image_package"])
#         src = f"./data/ani-pic/{vid}/oragin.mp4"     # 源文件路径
#         dst = os.path.join(traget_dir, f'{tar}.mp4')  # 目标文件路径
#         shutil.copy(src, dst)

#     src = f"./data/ani-pic/{todo}/oragin.png"     # 源文件路径
#     dst = os.path.join(traget_dir, 'image.png')  # 目标文件路径
#     shutil.copy(src, dst)




# trans_vido_dir = f'./facial/{todo}_trans'
# trans_img_s_dir = f'./facial/{todo}_s_trans'
# trans_img_l_dir = f'./facial/{todo}_l_trans'

# traget_dir = f'./package/{todo}'
# traget_video_dir = f'./package/{todo}/video_01'
# traget_img_l_dir = f'./package/{todo}/image_l'
# traget_img_s_dir = f'./package/{todo}/image_s'


# copy_specified_folders_with_structure(trans_vido_dir, traget_video_dir, ['Difference_R', "resized_image_package"])
# # copy_specified_folders_with_structure(trans_vido_dir, traget_dir, ['cropped_dir'])
# copy_specified_folders_with_structure(trans_img_s_dir, traget_img_s_dir, ['Difference_R', "resized_image_package"])
# copy_specified_folders_with_structure(trans_img_l_dir, traget_img_l_dir, ['Difference_R', "resized_image_package"])

# import shutil
# import os

# src = f"./data/ani-pic/{todo}/oragin.png"     # 源文件路径
# dst = os.path.join(traget_dir, 'image.png')  # 目标文件路径

# shutil.copy(src, dst)

# src = f"./data/ani-pic/{todo}/oragin.mp4"     # 源文件路径
# dst = os.path.join(traget_dir, 'video_01.mp4')  # 目标文件路径

# shutil.copy(src, dst)

# print("✅ 文件已复制到:", dst)