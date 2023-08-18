import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import time
import uuid

import torch
import random
import math
import imageio
from pathlib import Path
from PIL import Image


def main(args):
    # 定义有效的图片扩展名
    valid_extensions = {".png", ".jpg", ".jpeg"}

    input_folder = args.input_path
    filenames = os.listdir(input_folder)

    # 使用自定义的排序函数来处理帧序号
    # 假设所有的文件名都是以"frame"开头
    filenames.sort(key=lambda x: int(x[5:].split('.')[0]))

    # 读取文件夹内的图片
    images = []
    for filename in filenames:
        if os.path.splitext(filename)[1].lower() in valid_extensions:
            img = Image.open(os.path.join(input_folder, filename))
            img_resized = img.resize((args.size, args.size), Image.LANCZOS)
            images.append(img_resized)

    # 计算每一帧的duration
    frame_duration = int(args.duration * 1000 / len(images))  # 毫秒为单位

    # 创建GIF
    output_file = os.path.join(args.output_path, f'{uuid.uuid4()}.gif')
    images[0].save(output_file, save_all=True, append_images=images[1:], loop=args.loop, duration=frame_duration)

    print(f'Saved to {output_file}')


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
        default=256)
    parser.add_argument(
        '--duration',
        dest='duration',
        type=float,
        default=2)
    parser.add_argument(
        '--loop',
        dest='loop',
        type=bool,
        default=False)
    parser.add_argument(
        '--output_path',
        dest='output_path',
        type=str,
        default=None)

    main(parser.parse_args())
