import datetime
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import gradio as gr

import library.train_util as train_util
from library.custom_logging import setup_logging
from lora_gui import lora_tab

# Set up logging
log = setup_logging()

PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'


def on_images_uploaded(files):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('_workspace', current_time, 'original')
    os.makedirs(output_folder, exist_ok=True)

    # Move images to workspace
    for file in files:
        _, ext = os.path.splitext(file.name)
        ext = ext.lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            shutil.move(file.name, output_folder)

    return [
        output_folder,
        gr.update(value=f"`Images uploaded to: {output_folder}`")
    ]


def clear_upload_images(input_folder):
    if input_folder == '':
        return [
            '',
            gr.update(value=None),
            gr.update(value=f"`Nothing to clear`")
        ]

    dir_path = Path(input_folder)
    for item in dir_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    return [
        '',
        gr.update(value=None),
        gr.update(value=f"`Upload folder cleared`")
    ]


# Clear images for training
def clear_images(input_folder, preview_images_dict):
    if input_folder == '':
        return [
            gr.update(value=[]),
            gr.update(value=f"`Nothing to clear`")
        ]

    dir_path = Path(input_folder)
    for item in dir_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    preview_images_dict.clear()
    return [
        gr.update(value=[], visible=False),  # hide Gallery
        gr.update(value=f"`Data folder cleared`")
    ]


def process_images(
        input_folder,
        face_type,
        target_width,
        target_height,
        repeat,
        drop_threshold,
        preview_images_dict
        # progress=gr.Progress()
):
    if input_folder == '':
        return [
            "",
            gr.update(value=[], visible=False),  # hide Gallery
            gr.update(value=f"`Please upload images first!`"),
            gr.update(value=''),
            gr.update(value='')
        ]

    output_folder = f"{input_folder}/../processed/{repeat}_{face_type}"

    log.info(f'Processing images in {input_folder}...')
    # progress(0.0, desc="Start Processing")

    run_cmd = f'{PYTHON} "{os.path.join("DFL/mainscripts", "Extractor.py")}"'
    run_cmd += f' "{input_folder}"'
    run_cmd += f' "{output_folder}"'
    run_cmd += f' "{face_type}"'
    run_cmd += f' "{target_width}"'
    run_cmd += f' "{target_height}"'
    run_cmd += f' "100"'  # jpeg quality

    log.info(run_cmd)

    run_cmd2 = f'{PYTHON} "{os.path.join("DFL/mainscripts", "Sorter.py")}"'
    run_cmd2 += f' "{output_folder}"'
    run_cmd2 += f' "blur"'  # Sort method
    run_cmd2 += f' "{drop_threshold}"'  # Drop threshold

    def check_progress(process):
        count = 0
        while True:
            output = process.stdout.readline().decode()
            log.info(output)

            pattern = r'\[DONE\] (\d+)'
            match = re.search(pattern, output)
            if match:
                count = int(match.group(1))
                if count > 0:
                    log.info(f"[DONE] Detected faces: {count}")
                else:
                    log.error("No faces detected!")

            if process.poll() is not None:
                break
            time.sleep(0.5)
            yield None
        yield count

    def output():
        images = list(Path(output_folder).glob('*'))
        preview_images_dict.update({output_folder: images})
        return [
            f"{input_folder}/../processed",
            # show Gallery
            gr.update(value=[img for sublist in preview_images_dict.values() for img in sublist], visible=True),
            gr.update(value=f"`Face detection done, {face_type}`"),
            gr.update(value=f"{input_folder}/../processed"),
            gr.update(value=f"{input_folder}/../output")
        ]

    if os.name == 'posix':
        os.system(run_cmd)
        os.system(run_cmd2)
        return output()
    else:
        subprocess.run(run_cmd)
        subprocess.run(run_cmd2)
        # process = subprocess.Popen(
        #     run_cmd,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE
        # )
        # for _ in progress.tqdm(check_progress(), desc="Processing"):
        #     pass
        return output()


def _add_pre_postfix(
        folder: str = '',
        prefix: str = '',
        postfix: str = '',
        caption_file_ext: str = '.caption',
        recursive: bool = False
) -> None:
    """
    Add prefix and/or postfix to the content of caption files within a folder.
    If no caption files are found, create one with the requested prefix and/or postfix.

    Args:
        folder (str): Path to the folder containing caption files.
        prefix (str, optional): Prefix to add to the content of the caption files.
        postfix (str, optional): Postfix to add to the content of the caption files.
        caption_file_ext (str, optional): Extension of the caption files.
    """

    if prefix == '' and postfix == '':
        return

    # image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    # image_files = [
    #     f for f in os.listdir(folder) if f.lower().endswith(image_extensions)
    # ]
    image_files = train_util.glob_images_pathlib(Path(folder), recursive)

    for image_file in image_files:
        caption_file_name = os.path.splitext(image_file)[0] + caption_file_ext
        caption_file_path = caption_file_name

        if not os.path.exists(caption_file_path):
            with open(caption_file_path, 'w', encoding='utf8') as f:
                separator = ', ' if prefix and postfix else ''
                f.write(f'{prefix}{separator}{postfix}')
        else:
            with open(caption_file_path, 'r+', encoding='utf8') as f:
                content = f.read()
                content = content.rstrip()
                f.seek(0, 0)

                prefix_separator = ', ' if prefix else ''
                postfix_separator = ', ' if postfix else ''
                f.write(
                    f'{prefix}{prefix_separator}{content}{postfix_separator}{postfix}'
                )


def caption_images(
        train_data_dir,
        general_threshold,
        character_threshold,
        model,
        undesired_tags,
        prefix,
        postfix,
):
    # Check for images_dir_input
    if train_data_dir == '':
        return gr.update(value=f"`No training data!`")

    log.info(f'Captioning files in {train_data_dir}...')
    run_cmd = f'accelerate launch "./finetune/tag_images_by_wd14_tagger.py"'
    run_cmd += f' --batch_size=8'
    run_cmd += f' --general_threshold={general_threshold}'
    run_cmd += f' --character_threshold={character_threshold}'
    run_cmd += f' --caption_extension=".txt"'
    run_cmd += f' --model="{model}"'
    run_cmd += (
        f' --max_data_loader_n_workers=2'
    )

    run_cmd += f' --recursive'
    run_cmd += f' --remove_underscore'
    run_cmd += f' --frequency_tags'

    if not undesired_tags == '':
        run_cmd += f' --undesired_tags="{undesired_tags}"'
    run_cmd += f' "{train_data_dir}"'

    log.info(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)

    # Add prefix and postfix
    _add_pre_postfix(
        folder=train_data_dir,
        caption_file_ext='.txt',
        prefix=prefix,
        postfix=postfix,
        recursive=True
    )

    return gr.update(value='`Captioning done`')


# ref: gradio_wd14_caption_gui_tab
def _gradio_wd14_caption_gui(train_folder, info_text):
    with gr.Accordion('[Step 2] Captioning', open=False):
        with gr.Row():
            prefix = gr.Textbox(
                label='Prefix to add to WD14 caption',
                placeholder='(触发词)',
                interactive=True,
            )

            postfix = gr.Textbox(
                label='Postfix to add to WD14 caption',
                placeholder='(Optional)',
                interactive=True,
            )

        undesired_tags = gr.Textbox(
            label='Undesired tags',
            placeholder='(Optional) Separate `undesired_tags` with comma `(,)` if you want to remove multiple tags, e.g. `1girl,solo,smile`.',
            interactive=True,
        )

        # Model Settings
        with gr.Row():
            model = gr.Dropdown(
                label='Model',
                choices=[
                    'SmilingWolf/wd-v1-4-convnext-tagger-v2',
                    'SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
                    'SmilingWolf/wd-v1-4-vit-tagger-v2',
                    'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
                ],
                value='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
            )

            general_threshold = gr.Slider(
                value=0.35,
                label='General threshold',
                info='Adjust `general_threshold` for pruning tags (less tags, less flexible)',
                minimum=0,
                maximum=1,
                step=0.05,
            )
            character_threshold = gr.Slider(
                value=0.35,
                label='Character threshold',
                info='useful if you want to train with character',
                minimum=0,
                maximum=1,
                step=0.05,
            )

        caption_button = gr.Button('Caption images', variant='primary')

        caption_button.click(
            caption_images,
            inputs=[
                train_folder,
                general_threshold,
                character_threshold,
                model,
                undesired_tags,
                prefix,
                postfix,
            ],
            outputs=[info_text],
            show_progress=False,
            api_name='human_caption'
        )

config_file = 'presets/lora/lora_config.json'
def load_lora_config():
    with open(config_file) as f:
        return json.load(f)


def gradio_preprocess_images_gui_tab(headless=False):
    with gr.Tab('人像训练'):
        upload_folder = gr.Textbox(visible=False)
        train_folder = gr.Textbox(visible=False)
        preview_images_dict = gr.State({})

        info_text = gr.Markdown()

        with gr.Row():
            clear_upload_button = gr.Button('Clear uploaded images', variant='stop', scale=0)
            clear_train_button = gr.Button('Clear training folder', variant='stop', scale=0)

        with gr.Accordion('[Step 0] Select the image folder to upload'):
            with gr.Row(equal_height=False):
                upload_images = gr.File(
                    show_label=False,
                    file_count='directory',
                    # file_types=['image']
                )

        with gr.Row():
            images_preview = gr.Gallery(preview=True, columns=8, visible=False, scale=2)

        with gr.Accordion('[Step 1] Face detection and cropping', open=False):
            with gr.Row():
                face_type = gr.Dropdown(
                    label='Face detection type',
                    choices=[
                        'half_face',
                        'midfull_face',
                        'full_face',
                        'full_face_no_align',
                        'whole_face',
                        'head',
                        'head_no_align',
                    ],
                    value='full_face'
                )
                target_width = gr.Slider(label='Output width', value=512, minimum=256, maximum=2048, step=64)
                target_height = gr.Slider(label='Output height', value=768, minimum=256, maximum=2048, step=64,
                                          visible=False)
                repeat = gr.Slider(label='Repeat per image when training', value=6, minimum=2, maximum=20, step=1)
                drop_threshold = gr.Slider(label='Discard blurry images', value=0.0, minimum=0, maximum=1,
                                           step=0.05)

            with gr.Row():
                submit_images_button = gr.Button('Process images', variant='primary')

        _gradio_wd14_caption_gui(train_folder, info_text)

        with gr.Accordion('[Step 3] Train', open=False):
            with gr.Accordion('Params', open=False):
                lora_config_json = gr.JSON(load_lora_config, show_label=False)

            (
                lora_train_data_dir,
                lora_reg_data_dir,
                lora_output_dir,
                lora_logging_dir,
            ) = lora_tab(headless=True)

        # Event listeners
        upload_images.upload(
            on_images_uploaded,
            inputs=[upload_images],  # files
            outputs=[upload_folder, info_text],
        )

        submit_images_button.click(
            process_images,
            inputs=[
                upload_folder,
                face_type,
                target_width,
                target_height,
                repeat,
                drop_threshold,
                preview_images_dict,
            ],
            outputs=[
                train_folder,
                images_preview,
                info_text,
                lora_train_data_dir,
                lora_output_dir,
            ],
            show_progress=False,
            api_name='human_preprocess'
        )

        clear_upload_button.click(
            clear_upload_images,
            inputs=[upload_folder],
            outputs=[
                upload_folder,
                upload_images,
                info_text,
            ],
            show_progress=False
        )
        clear_train_button.click(
            clear_images,
            inputs=[train_folder, preview_images_dict],
            outputs=[
                images_preview,
                info_text,
            ],
            show_progress=False
        )
