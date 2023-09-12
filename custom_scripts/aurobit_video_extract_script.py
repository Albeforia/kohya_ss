import argparse
import glob
import os
import re
import shutil
import subprocess

import cv2
import imageio
from PIL import Image


def get_matting_cmd(from_folder, to_folder):
    run_cmd = f'accelerate launch "{os.path.join("Matting/tools", "predict.py")}"'
    run_cmd += f' "--config={os.path.join("Matting/configs/ppmattingv2", "ppmattingv2-stdc1-human_512.yml")}"'
    run_cmd += f' "--model_path={os.path.join("Matting/pretrained_models", "ppmattingv2-stdc1-human_512.pdparams")}"'
    run_cmd += f' "--image_path={from_folder}"'
    run_cmd += f' "--save_dir={to_folder}"'
    run_cmd += f' "--fg_estimate=True"'
    run_cmd += f' "--background=w"'  # Add white background
    run_cmd += f' "--save_mask=True"'

    return run_cmd


def _rename_files(directory):
    file_list = os.listdir(directory)
    for filename in file_list:
        old_path = os.path.join(directory, filename)
        if os.path.isfile(old_path):
            match = re.match(r'frame(\d+)_(\w+)\.(jpg|png)', filename)
            if match:
                new_filename = f"frame{match.group(1)}.{match.group(3)}"
                new_path = os.path.join(directory, new_filename)
                os.rename(old_path, new_path)


def main(args):
    target_size = args.size
    reader = imageio.get_reader(args.input_path)

    print('Extracting frames...')

    counter = 0
    for i, frame in enumerate(reader):
        height, width, _ = frame.shape

        # 计算要缩放的新宽度和高度，保持宽高比不变
        if height < width:
            new_height = target_size
            new_width = int(target_size * width / height)
        else:
            new_width = target_size
            new_height = int(target_size * height / width)

        # 使用 PIL 对帧进行缩放
        frame = Image.fromarray(frame)
        frame = frame.resize((new_width, new_height), Image.LANCZOS)

        # 计算居中裁剪的起始点
        start_y = (new_height - target_size) // 2
        start_x = (new_width - target_size) // 2

        # 对帧进行居中裁剪
        frame = frame.crop((start_x, start_y, start_x + target_size, start_y + target_size))

        # 保存为图片
        frame.save(os.path.join(args.output_path, f'frame{counter}.jpg'))

        counter += 1

    print(f'Extracted {counter} frames')

    # Matting
    mask_path = os.path.join(args.output_path, '..', 'mask')
    run_cmd = get_matting_cmd(args.output_path, mask_path)
    subprocess.run(run_cmd, shell=True)

    matted_path = os.path.join(args.output_path, '..', 'matted')
    os.makedirs(matted_path, exist_ok=True)
    for file_path in glob.glob(os.path.join(mask_path, '*_rgba.*')):
        shutil.move(file_path, matted_path)

    _rename_files(mask_path)
    _rename_files(matted_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_path',
        dest='input_path',
        type=str,
        default=None)
    parser.add_argument(
        '--size',
        dest='size',
        type=int,
        default=512)
    parser.add_argument(
        '--output_path',
        dest='output_path',
        type=str,
        default=None)

    main(parser.parse_args())
