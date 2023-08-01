import datetime
import json
import os
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import gradio as gr

import library.train_util as train_util
from library.custom_logging import setup_logging
from lora_gui import lora_tab, train_model

# Set up logging
log = setup_logging()

PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'

config_file = 'presets/lora/user_presets/lora_config.json'


def get_matting_cmd(from_folder, to_folder):
    run_cmd = f'{PYTHON} "{os.path.join("Matting/tools", "predict.py")}"'
    run_cmd += f' "--config={os.path.join("Matting/configs/ppmattingv2", "ppmattingv2-stdc1-human_512.yml")}"'
    run_cmd += f' "--model_path={os.path.join("Matting/pretrained_models", "ppmattingv2-stdc1-human_512.pdparams")}"'
    run_cmd += f' "--image_path={from_folder}"'
    run_cmd += f' "--save_dir={to_folder}"'
    run_cmd += f' "--fg_estimate=True"'
    run_cmd += f' "--background=w"'  # Add white background

    return run_cmd


def on_images_uploaded(
        files,
        auto_matting,
        lora_config_json,
        api_call=False
):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('_workspace', current_time, 'original')
    final_output_folder = f"{output_folder}/../output"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(final_output_folder)
    lora_config_json.update({'logging_dir': os.path.join(final_output_folder, 'train_log')})
    lora_config_json.update({'output_dir': f"{final_output_folder}"})

    # Move images to workspace
    for file in files:
        filepath = file if isinstance(file, str) else file.name
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            shutil.move(filepath, output_folder)

    # Matting
    if auto_matting:
        tmp_folder = f"{output_folder}_"
        os.rename(output_folder, tmp_folder)
        run_cmd = get_matting_cmd(tmp_folder, output_folder)

        log.info(run_cmd)
        # if os.name == 'posix':
        #     os.system(run_cmd)
        # else:
        if api_call:
            with open(f"{final_output_folder}/log.txt", 'a') as f:
                subprocess.run(run_cmd, stdout=f, stderr=subprocess.STDOUT)
        else:
            subprocess.run(run_cmd)

    return [
        output_folder,
        gr.update(value=f"`Images uploaded to: {output_folder}`"),
        gr.update(value=config_file),  # config_file_name
        lora_config_json,
    ]


def clear_upload_images(input_folder):
    if input_folder == '':
        return [
            '',
            gr.update(value=None),
            gr.update(value=f"`Nothing to clear`")
        ]

    dir_path = Path(input_folder)
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path = Path(f"{dir_path}_")
    if dir_path.exists():
        shutil.rmtree(dir_path)
    # for item in dir_path.iterdir():
    #     if item.is_dir():
    #         shutil.rmtree(item)
    #     else:
    #         item.unlink()
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
        pass_ratio,
        preview_images_dict,
        lora_config_json,
        api_call=False,
        # progress=gr.Progress()
):
    if input_folder == '':
        return [
            "",
            gr.update(value=[], visible=False),  # hide Gallery
            gr.update(value=f"`Please upload images first!`"),
            lora_config_json,
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
    run_cmd += f' "{pass_ratio}"'

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
        lora_config_json.update({'train_data_dir': f"{input_folder}/../processed"})
        with open(config_file, 'w') as json_file:
            json.dump(lora_config_json, json_file)
        return [
            f"{input_folder}/../processed",
            # show Gallery
            gr.update(value=[img for sublist in preview_images_dict.values() for img in sublist], visible=True),
            gr.update(value=f"`Face detection done, {face_type}`"),
            lora_config_json,
        ]

    # if os.name == 'posix':
    #     os.system(run_cmd)
    #     os.system(run_cmd2)
    #     return output()
    # else:
    if api_call:
        with open(f"{lora_config_json['output_dir']}/log.txt", 'a') as f:
            subprocess.run(run_cmd, stdout=f, stderr=subprocess.STDOUT)
            subprocess.run(run_cmd2, stdout=f, stderr=subprocess.STDOUT)
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
        api_call=False,
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
    # if os.name == 'posix':
    #     os.system(run_cmd)
    # else:
    if api_call:
        with open(f"{train_data_dir}/../output/log.txt", 'a') as f:
            subprocess.run(run_cmd, stdout=f, stderr=subprocess.STDOUT)
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
        )


def load_lora_config():
    with open(config_file) as f:
        return json.load(f)


def get_file_paths(directory):
    file_paths = []
    for file in os.listdir(directory):
        full_file_path = os.path.join(directory, file)
        if os.path.isfile(full_file_path):
            file_paths.append(full_file_path)
    return file_paths


def _train_api(input_folder, model_path, trigger_words):
    config = load_lora_config()
    config.update({'pretrained_model_name_or_path': model_path})

    try:
        uploaded_files = get_file_paths(input_folder)

        # Preprocess
        work_folder, _, _, _ = on_images_uploaded(uploaded_files, auto_matting=True, lora_config_json=config,
                                                  api_call=True)
        train_folder, _, _, _ = \
            process_images(work_folder, 'half_face', 512, 768, 10, 0, 1, {}, config, api_call=True)
        process_images(work_folder, 'head', 512, 768, 10, 0, 1, {}, config, api_call=True)
        # ...

        # Caption
        caption_images(
            train_folder,
            0.4,
            0.4,
            'SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
            '',
            trigger_words,
            '',
            api_call=True
        )

        log.info(config)
        # Train
        config.pop('stop_text_encoder_training')
        config['stop_text_encoder_training_pct'] = 0  # Not yet supported
        train_model(headless=True, print_only=True, **config)
        with open(f"{config['output_dir']}/log.txt", 'a') as f:
            f.write('SUCCESS')

        return os.path.abspath(config['output_dir'])
    except Exception as e:
        stack = traceback.format_exc()
        print(stack)
        with open(f"{config['output_dir']}/log.txt", 'a') as f:
            f.write(stack)

        return os.path.abspath(config['output_dir'])


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
                    # file_types=['image'],
                    sacle=2
                )
                auto_matting = gr.Checkbox(value=True, label='Auto matting', scale=0)

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
                drop_threshold = gr.Slider(label='Enhance blurry faces', value=0.0, minimum=0, maximum=1, step=0.05,
                                           info='Recommend 0.3 for head; 0.2 for full face', visible=False)
                pass_ratio = gr.Slider(label='Pass ratio', value=1.0, minimum=0, maximum=1, step=0.1,
                                       info='Only this ratio of images will proceed to cropping')

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
                # new
                config_file_name,
            ) = lora_tab(headless=True)

        # Event listeners
        upload_images.upload(
            on_images_uploaded,
            inputs=[
                upload_images,  # files
                auto_matting,
                lora_config_json
            ],
            outputs=[
                upload_folder,
                info_text,
                config_file_name,
                lora_config_json
            ],
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
                pass_ratio,
                preview_images_dict,
                lora_config_json,
            ],
            outputs=[
                train_folder,
                images_preview,
                info_text,
                # lora_train_data_dir,
                # lora_output_dir,
                lora_config_json,
            ],
            show_progress=False,
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

        # API
        api_only_image_path = gr.Textbox('', visible=False)
        api_only_model_path = gr.Textbox('', visible=False)
        api_only_output_path = gr.Textbox('', visible=False)
        api_only_trigger_words = gr.Textbox('', visible=False)
        api_only_train = gr.Button('', visible=False)
        api_only_train.click(
            _train_api,
            inputs=[api_only_image_path, api_only_model_path, api_only_trigger_words],
            outputs=[api_only_output_path],
            api_name='train_lora'
        )
