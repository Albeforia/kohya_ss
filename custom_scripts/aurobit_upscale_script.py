import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import time
import torch
import random
import math
from pathlib import Path
from PIL import Image
from RealESRGAN import RealESRGAN


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RealESRGAN(device, scale=args.scale)

    model_name = 'RealESRGAN/weights/RealESRGAN_x2.pth'
    if args.scale == 4:
        model_name = 'RealESRGAN/weights/RealESRGAN_x4.pth'
    elif args.scale == 8:
        model_name = 'RealESRGAN/weights/RealESRGAN_x8.pth'

    # Download weights if not cached
    model.load_weights(model_name, download=True)

    path_to_image = args.input_path
    image = Image.open(path_to_image).convert('RGB')

    # Upscale
    sr_image = model.predict(image)

    img_filename = os.path.basename(path_to_image)
    fname, fext = os.path.splitext(img_filename)
    sr_img_filename = f"{fname}_x{args.scale}{fext}"

    print(path_to_image)

    image.save(os.path.join(args.output_path, img_filename))
    sr_image.save(os.path.join(args.output_path, sr_img_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_path',
        dest='input_path',
        type=str,
        default=None)
    parser.add_argument(
        '--scale',
        dest='scale',
        type=int,
        default=2)
    parser.add_argument(
        '--output_path',
        dest='output_path',
        type=str,
        default=None)

    main(parser.parse_args())
