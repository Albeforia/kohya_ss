import argparse
import base64
import datetime
import glob
import json
import os
import io
import re
import shutil
import subprocess
import time
import torch
import random
import requests
import math
import imageio
from pathlib import Path
from PIL import Image


def list_cn(url):
    cn_url = url + "/controlnet/control_types"
    r = requests.get(url=cn_url)
    if r.status_code == 200:
        return r.json()

    print(f"Error occurred when querying controlnet, CODE {r.status_code}")
    return None


def img_str(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def call_sd(url, payload, frame_idx, frame_map, cn_config, input_img=None):
    # For img2img
    if input_img is not None:
        input_img_str = img_str(Image.open(input_img))
        payload['init_images'] = [input_img_str]

    # Setup CNs
    cn_unit = 0
    if cn_config['count'] > 0:
        for cn in payload['alwayson_scripts']['controlnet']['args']:
            cn_img_path = frame_map[cn_config['source'][cn_unit]][frame_idx]
            cn['input_image'] = img_str(Image.open(cn_img_path))
            cn_unit += 1

    # Send to SD
    response = requests.post(url=url, json=payload)
    if response.status_code == 200:
        try:
            r = response.json()
            i = r['images'][0]
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
            info = json.loads(r['info'])
            return image, info['seed']
        except Exception as e:
            print(f"Error occurred when parsing result: {e}")
            return None, -1
    else:
        print(f"Error occurred when calling {url}, CODE {response.status_code}")
        return None, -1


def get_frame_map(work_path):
    source_folder = os.path.join(work_path, 'sliced')
    mask_folder = os.path.join(work_path, 'mask')
    matted_folder = os.path.join(work_path, 'matted')

    source_files = os.listdir(source_folder)
    mask_files = os.listdir(mask_folder)
    matted_files = os.listdir(matted_folder)

    source_files.sort(key=lambda x: int(re.search(r'frame(\d+)', x).group(1)))
    mask_files.sort(key=lambda x: int(re.search(r'frame(\d+)', x).group(1)))
    matted_files.sort(key=lambda x: int(re.search(r'frame(\d+)', x).group(1)))

    return {
        'source': [os.path.join(source_folder, f) for f in source_files],
        'mask': [os.path.join(mask_folder, f) for f in mask_files],
        'matted': [os.path.join(matted_folder, f) for f in matted_files],
    }


def main(args):
    end_point = '/sdapi/v1/txt2img' if args.mode == 'txt2img' else '/sdapi/v1/img2img'
    url = f"{args.api_addr}{end_point}"

    frame_map = get_frame_map(args.work_path)

    # cn_list = list_cn(args.api_addr)
    # print(cn_list)

    config_file = f"custom_scripts/sd_templates/{args.params_file}"
    with open(config_file) as f:
        config = json.load(f)
        params = config['payload']

    # Set prompts
    params['prompt'] = args.prompt

    # The first frame
    img, seed = call_sd(url, params, 0, frame_map, config['cn_config'], input_img=None)
    params['seed'] = seed
    if img is not None:
        img.save(os.path.join(args.output_path, os.path.basename(frame_map['source'][0])))

        # Remaining frames
        for idx in range(1, len(frame_map['source'])):
            img, _ = call_sd(url, params, idx, frame_map, config['cn_config'], input_img=None)
            if img is not None:
                img.save(os.path.join(args.output_path, os.path.basename(frame_map['source'][idx])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--work_path',
        dest='work_path',
        type=str,
        default='')
    parser.add_argument(
        '--api_addr',
        dest='api_addr',
        type=str,
        default='')
    parser.add_argument(
        '--mode',
        dest='mode',
        type=str,
        default='txt2img')
    parser.add_argument(
        '--params_file',
        dest='params_file',
        type=str,
        default='')
    parser.add_argument(
        '--prompt',
        dest='prompt',
        type=str,
        default='')
    parser.add_argument(
        '--output_path',
        dest='output_path',
        type=str,
        default=None)

    main(parser.parse_args())
