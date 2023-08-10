import datetime
import json
import os
import re
import requests
import shutil
import subprocess
import time
import traceback
import random
import math
import uuid
from PIL import Image
from pathlib import Path

import gradio as gr

import library.train_util as train_util
from library.custom_logging import setup_logging
from lora_gui import lora_tab, train_model

# Set up logging
log = setup_logging()

templates_root = 'LightTemplates/Images'


def get_templates():
    if not os.path.exists(templates_root):
        return []
    l = [name for name in os.listdir(templates_root) if os.path.isdir(os.path.join(templates_root, name))]
    return sorted(l)


def on_template_select(evt: gr.SelectData):
    # log.info(f"You selected {evt.value} at {evt.index} from {evt.target}")
    img_file = os.path.join(templates_root, evt.value, 'LightingImage.webp')
    return [
        gr.update(value=img_file),  # preview
        evt.value,  # template name
    ]


def get_content_type(filename):
    ext = os.path.splitext(filename)[1]
    if ext == '.png':
        return 'image/png'
    elif ext == '.jpg' or ext == '.jpeg':
        return 'image/jpeg'
    elif ext == '.webp':
        return 'image/webp'
    else:
        return 'application/octet-stream'


def composite_image(foreground_path, background_path):
    foreground = Image.open(foreground_path).convert("RGBA")
    background = Image.open(background_path).convert("RGBA")

    new_height = foreground.height
    new_width = int(new_height * background.width / background.height)

    background = background.resize((new_width, new_height), Image.LANCZOS)

    left = round((background.width - foreground.width) / 2)
    right = left + foreground.width

    background = background.crop((left, 0, right, new_height))

    return Image.alpha_composite(background, foreground)


def download_image(session, url, save_path):
    _, image_extension = os.path.splitext(url)
    save_name = f'{uuid.uuid4()}{image_extension}'

    run_cmd = f'accelerate launch "{os.path.join("LightTemplates", "download_image.py")}"'
    run_cmd += f' "--url={url}"'
    run_cmd += f' "--save_path={save_path}"'
    run_cmd += f' "--save_name={save_name}"'
    time.sleep(5)  # Must wait a while otherwise downloading would fail
    log.info(run_cmd)
    p = subprocess.run(run_cmd, shell=True)
    if p.returncode != 0:
        return None
    return os.path.join(save_path, save_name)


def process_relit(token, source_image, selected_template):
    if source_image is None or selected_template == '':
        return [
            gr.update(value='Please confirm template and image are set!', visible=True),
            gr.update(value=None, visible=False)
        ]

    relight_params = {
        "is_hdri_lighting": False, "is_keyframe": False, "auto_scale": False, "resolution": "high",
        "aspect_ratio": "Original", "use_src_alpha": False, "lighting_strength": 1,
        "lighting_rotation_theta": 90, "lighting_rotation_phi": 180
    }

    source_image_path = source_image
    lighting_image_path = os.path.join(templates_root, selected_template, 'LightingImage.webp')
    background_image_path = os.path.join(templates_root, selected_template, 'BackgroundImage.webp')

    source_image_file = open(source_image_path, 'rb')
    lighting_image_file = open(lighting_image_path, 'rb')
    background_image_file = open(background_image_path, 'rb')

    session = requests.Session()
    url = "https://api.beeble.ai/RelightImage"
    payload = {
        'relight_params': json.dumps(relight_params)
    }
    files = [
        ('source_image', ('source.webp', source_image_file, get_content_type(source_image_path))),
        ('lighting_image', ('lighting.webp', lighting_image_file, get_content_type(lighting_image_path))),
        ('background_image', ('background.webp', background_image_file, get_content_type(background_image_path)))
    ]
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
        'Dnt': '1',
        'Origin': 'https://www.switchlight.beeble.ai',
        'Referer': 'https://www.switchlight.beeble.ai/',
        'Sec-Ch-Ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }

    response = session.request("POST", url, headers=headers, data=payload, files=files)

    source_image_file.close()
    lighting_image_file.close()
    background_image_file.close()

    if response.status_code == 200:
        rdict = json.loads(response.text)
        image_url = rdict.get('relit_foreground')
        saved_file = download_image(session, image_url, save_path='_workspace/relit')
        if saved_file is not None:
            final_image = composite_image(saved_file, background_image_path)
            return [
                gr.update(value='', visible=False),
                gr.update(value=final_image, visible=True)
            ]
        else:
            return [
                gr.update(value='Failed to download, you could try again', visible=True),
                gr.update(value=None, visible=False)
            ]
    else:
        return [
            gr.update(value=f'Relighting failed: {response.status_code} {response.text}', visible=True),
            gr.update(value=None, visible=False)
        ]


def gradio_aurobit_relighting_gui_tab(headless=False):
    with gr.Tab('光照迁移'):
        selected_template = gr.Textbox('', visible=False)

        with gr.Row():
            with gr.Column(scale=2):
                token = gr.Textbox(
                    label='0. Set a valid token',
                )
                select_template = gr.Dropdown(
                    label='1. Choose a template',
                    choices=get_templates(),
                )
                source_image = gr.Image(
                    label='2. Upload an image',
                    type='filepath',
                )

                process_button = gr.Button('Transfer lighting', variant='primary')

            template_preview = gr.Image(
                label='Template preview',
                interactive=False,
                scale=1
            )

            result_preview = gr.Image(
                show_label=False,
                visible=False,
                interactive=False,
                type='pil'
            )

        with gr.Row():
            tmp_text = gr.Textbox(
                show_label=False
            )

        select_template.select(
            on_template_select,
            inputs=[],
            outputs=[template_preview, selected_template]
        )

        process_button.click(
            process_relit,
            inputs=[
                token, source_image, selected_template
            ],
            outputs=[
                tmp_text,
                result_preview
            ]
        )
