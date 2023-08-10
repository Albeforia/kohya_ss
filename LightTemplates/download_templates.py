import os
import requests
from PIL import Image

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()  # 如果响应的状态码不是200，引发HTTPError异常
    except requests.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return False
    except Exception as err:
        print(f'Other error occurred: {err}')
        return False
    else:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        return True

def check_images():
    root_dir = 'Images'
    for dir_name, _, file_list in os.walk(root_dir):
        if dir_name == root_dir:
            continue

        if 'BackgroundImage.webp' not in file_list or 'LightingImage.webp' not in file_list:
            print(f'Error: Missing image in {dir_name}')
            continue

        for image_file in ['BackgroundImage.webp', 'LightingImage.webp']:
            image_path = os.path.join(dir_name, image_file)
            try:
                with Image.open(image_path) as img:
                    img.verify()  # 验证图片
            except (IOError, SyntaxError) as e:
                print(f'Error: Cannot open image at {image_path}, {e}')

def main():
    # 下载前请删除Images文件夹
    # 请求API
    try:
        response = requests.get('https://api.beeble.ai/Template?query=&image_type=sRGB', timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        # 确保主目录存在
        if not os.path.exists('Images'):
            os.makedirs('Images')

        for item in data:
            # 创建标题目录
            title = item['Title']
            folder_path = os.path.join('Images', title)
            
            # 如果目录名已存在，添加后缀
            i = 1
            while os.path.exists(folder_path):
                folder_path = os.path.join('Images', title + '_' + str(i))
                i += 1

            os.makedirs(folder_path)
            
            # 下载并保存BackgroundImage
            bg_image_url = item['BackgroundImage']
            if not download_image(bg_image_url, os.path.join(folder_path, 'BackgroundImage.webp')):
                print(f'Failed to download BackgroundImage from {bg_image_url}')
            
            # 下载并保存LightingImage
            light_image_url = item['LightingImage']
            if not download_image(light_image_url, os.path.join(folder_path, 'LightingImage.webp')):
                print(f'Failed to download LightingImage from {light_image_url}')

    check_images()


if __name__ == "__main__":
    main()