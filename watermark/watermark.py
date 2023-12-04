import argparse
import math
import os

from PIL import Image, ImageDraw, ImageFont


def _add_watermark(params, img):
    """
    create watermark onto the original image
    """
    text_color = params['text_color']

    # input image
    img = img.convert('RGBA')
    img_size = img.size

    # watermark image
    s_ext = int(math.sqrt(img_size[0] * img_size[0] + img_size[1] * img_size[1]))
    mark_img = Image.new(mode='RGBA', size=(s_ext, s_ext), color=(0, 0, 0, 0))
    draw_table = ImageDraw.Draw(im=mark_img)

    # text
    text = params['text']
    font = ImageFont.truetype(params['font'], params['font_size'])
    # text_width, text_height = draw_table.textsize(text, font=font)
    _, _, text_width, text_height = draw_table.textbbox((0, 0), text, font=font)

    # padding
    text_width = text_width * 2  # padding
    text_height = text_height * 2

    # draw text
    nx = int(s_ext / text_width) + 2
    ny = int(s_ext / text_height) + 2
    x_offset = [0, -text_width / 2]
    for xi in range(nx):
        for yi in range(ny):
            px = xi * text_width + x_offset[yi % 2]
            py = yi * text_height

            draw_table.text((px, py), text, font=font,
                            fill=(text_color[0], text_color[1], text_color[2], text_color[3]))

    # rotate watermark image
    mark_img = mark_img.rotate(params['rotate_angle'])

    # crop watermark image
    left = int((s_ext - img_size[0]) / 2)
    right = left + img_size[0]
    top = int((s_ext - img_size[1]) / 2)
    bottom = top + img_size[1]
    mark_img = mark_img.crop((left, top, right, bottom))

    # alpha composite
    img = Image.alpha_composite(img, mark_img)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', dest='input_path', type=str, default='')
    parser.add_argument('--output_path', dest='output_path', type=str, default='')
    args = parser.parse_args()

    default_params = {
        "image": None,  # Image object or list of Image objects
        "text": "PikaPiku",
        "font": "watermark/arial.ttf",
        "font_size": 40,
        "text_color": [255, 255, 255, 25],
        "rotate_angle": 45
    }

    print(f"Add watermark to {args.input_path}")

    img = Image.open(args.input_path)
    output = _add_watermark(default_params, img)

    image_name, image_ext = os.path.splitext(args.input_path)
    output_image_path = image_name + '_w' + image_ext
    output.save(output_image_path)


if __name__ == "__main__":
    main()
