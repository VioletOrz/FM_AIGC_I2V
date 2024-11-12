##########################################################################################
#默认生成的图包格式为字典序，使用这个工具可以将图片名字从img_001的字典序格式转换为img_1的数值形式
#rename_images_in_folder_set_deep 是只重命名指定目录下的某一层级的图片
#rename_images_in_folder 会重命名这个目录所有子目录下的图片，小心使用
##########################################################################################
import os

def rename_images_in_folder_set_deep(folder_path, deep):
    # 遍历所有deep级目录
    for root, dirs, files in os.walk(folder_path):
        if len(os.path.relpath(root, folder_path).split(os.sep)) == deep:
            for file_name in files:
                # 检查文件名是否匹配 img_001 ~ img_100 的格式
                if file_name.startswith("img_") and file_name[4:7].isdigit() and 1 <= int(file_name[4:7]) <= 100:
                    # 去掉前导零并生成新文件名
                    new_name = f"img_{int(file_name[4:7])}{file_name[7:]}"
                    old_path = os.path.join(root, file_name)
                    new_path = os.path.join(root, new_name)
                    # 重命名文件
                    os.rename(old_path, new_path)
                    print(f"Renamed {old_path} to {new_path}")

def rename_images_in_folder(folder_path):
    # 遍历所有子目录和文件
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # 检查文件名是否匹配 img_001 ~ img_999 的格式
            if file_name.startswith("img_") and file_name[4:7].isdigit() and 1 <= int(file_name[4:7]) <= 999:
                # 去掉前导零并生成新文件名
                new_name = f"img_{int(file_name[4:7])}{file_name[7:]}"
                old_path = os.path.join(root, file_name)
                new_path = os.path.join(root, new_name)
                # 重命名文件
                os.rename(old_path, new_path)
                print(f"Renamed {old_path} to {new_path}")

# 使用指定的文件夹路径
if __name__ == "__main__":
    #folder_path = "你的文件夹路径"
    #rename_images_in_folder(folder_path)

    # 使用指定的文件夹路径
    folder_path = r"C:\Users\Violet\Desktop\新建文件夹"
    deep = 3 #图片所处的目录深度
    #rename_images_in_folder_set_deep(folder_path, deep)
    rename_images_in_folder(folder_path)