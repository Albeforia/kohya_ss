import argparse
import os

import torch
from PIL import Image

from RealESRGAN import RealESRGAN


def _get_image_file_paths(directory):
    image_file_paths = []
    for file in os.listdir(directory):
        full_file_path = os.path.join(directory, file)
        if os.path.isfile(full_file_path):
            # Check the file extension
            _, ext = os.path.splitext(full_file_path)
            if ext.lower() in ['.jpg', '.jpeg', '.png']:
                image_file_paths.append(full_file_path)
    return image_file_paths


def upscale_single_image(model, image_file, scale, output_path):
    image = Image.open(image_file).convert('RGB')

    # Upscale
    sr_image = model.predict(image)

    img_filename = os.path.basename(image_file)
    fname, fext = os.path.splitext(img_filename)
    sr_img_filename = f"{fname}_x{scale}{fext}"

    # image.save(os.path.join(output_path, img_filename))
    sr_image.save(os.path.join(output_path, sr_img_filename))


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

    if os.path.isdir(args.input_path):
        files = _get_image_file_paths(args.input_path)
        for img_file in files:
            upscale_single_image(model, img_file, args.scale, args.output_path)
    elif os.path.isfile(args.input_path):
        path_to_image = args.input_path
        upscale_single_image(model, path_to_image, args.scale, args.output_path)


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
