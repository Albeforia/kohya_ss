import datetime
import json
import math
import os
import random
import re
import shutil
import subprocess
import timeit
import traceback
import asyncio
from pathlib import Path

import gradio as gr
from PIL import Image

import library.train_util as train_util
from library.aurobit_upscale_gui import gradio_aurobit_upscale_gui_tab
from library.aurobit_video_gui import gradio_aurobit_video_gui_tab
from library.custom_logging import setup_logging
from lora_gui import lora_tab, train_model

# Set up logging
log = setup_logging()


# PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'


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


def run_cmd_with_log(cmd, api_call, log_file):
    log.info(cmd)
    if api_call:
        with open(log_file, 'a') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)
    else:
        subprocess.run(cmd, shell=True)


def get_matting_cmd(from_folder, to_folder):
    run_cmd = f'accelerate launch "{os.path.join("Matting/tools", "predict.py")}"'
    run_cmd += f' "--config={os.path.join("Matting/configs/ppmattingv2", "ppmattingv2-stdc1-human_512.yml")}"'
    run_cmd += f' "--model_path={os.path.join("Matting/pretrained_models", "ppmattingv2-stdc1-human_512.pdparams")}"'
    run_cmd += f' "--image_path={from_folder}"'
    run_cmd += f' "--save_dir={to_folder}"'
    run_cmd += f' "--fg_estimate=True"'
    run_cmd += f' "--background=w"'  # Add white background

    return run_cmd


def on_images_uploaded_simple(files):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('_workspace', 'temp', current_time)
    os.makedirs(output_folder)

    # Move images to workspace
    for file in files:
        filepath = file if isinstance(file, str) else file.name
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            shutil.move(filepath, output_folder)

    return output_folder


def compute_max_epochs(N):
    if N <= 20:
        return 10
    elif 20 < N <= 40:
        return 10 + (N - 20) // 2
    else:
        return 20


def on_images_uploaded(
        files,
        auto_matting,
        auto_upscale,
        api_call=False
):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('_workspace', current_time, 'original')
    final_output_folder = f"{output_folder}/../output"
    log_file = f"{final_output_folder}/log.txt"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(final_output_folder)

    # Move images to workspace
    for file in files:
        filepath = file if isinstance(file, str) else file.name
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            shutil.move(filepath, output_folder)

    # Analysis
    run_cmd0 = f'accelerate launch "{os.path.join("custom_scripts", "aurobit_face_analysis_script.py")}"'
    run_cmd0 += f' "--input_path={output_folder}"'
    run_cmd0 += f' "--detect_mode=retinaface"'
    run_cmd0 += f' "--output_path={final_output_folder}"'
    run_cmd_with_log(run_cmd0, api_call, log_file)
    with open(f'{final_output_folder}/faces.txt') as f:
        j = json.load(f)
        detected_faces = j['faces']
        analysis_result = j['stats']
        invalid_images = j['invalid']

    # Delete invalid images
    for invalid_img in invalid_images:
        os.remove(invalid_img)

    # Compute training epochs according to faces count
    max_epochs = compute_max_epochs(len(detected_faces))

    if auto_upscale:
        to_upscale = set()
        for face_data in detected_faces:
            # TODO Hard-coded threshold
            if max(face_data['size']) < 225:
                to_upscale.add(face_data['source'])

        if len(to_upscale) > 0:
            tmp_folder = os.path.join(output_folder, '..', 'temp')
            os.makedirs(tmp_folder, exist_ok=True)
            for f in to_upscale:
                shutil.move(f, tmp_folder)
            # run_cmdx = f'accelerate launch "{os.path.join("custom_scripts", "aurobit_upscale_script.py")}"'
            # run_cmdx += f' "--input_path={tmp_folder}"'
            # run_cmdx += f' "--scale=2"'
            # run_cmdx += f' "--output_path={output_folder}"'
            # Use codeformer instead (slower)
            run_cmdx = f'accelerate launch "{os.path.join("CodeFormer", "inference_codeformer.py")}"'
            run_cmdx += f' "--input_path={tmp_folder}"'
            run_cmdx += f' "--fidelity_weight=0.85"'
            run_cmdx += f' "--output_path={tmp_folder}"'
            run_cmdx += f' "--bg_upsampler=realesrgan"'
            run_cmdx += f' "--face_upsample"'
            run_cmd_with_log(run_cmdx, api_call, log_file)
            for f in os.listdir(os.path.join(tmp_folder, 'final_results')):
                shutil.move(os.path.join(tmp_folder, 'final_results', f), output_folder)
            # shutil.rmtree(tmp_folder)

    # Update folders for lora config
    with open('training_profile.json') as f:
        training_profile = json.load(f)
    basemodel_type = training_profile.get('default', 'majic')
    basemodel_profile = training_profile[basemodel_type]

    mode = 'female'
    if analysis_result['most_common_gender'] == 'Man':
        mode = 'male'
    config_file_path = basemodel_profile[mode]

    lora_config_json = load_lora_config(use_wandb=not api_call, mode=mode, path=config_file_path)
    lora_config_json.update({'train_data_dir': f"{output_folder}/../processed"})
    lora_config_json.update({'output_dir': f"{final_output_folder}"})
    lora_config_json.update({'logging_dir': os.path.join(final_output_folder, 'train_log')})
    lora_config_json.update({'pretrained_model_name_or_path': basemodel_profile['path']})
    lora_config_json.update({'epoch': max_epochs})
    with open(config_file_path, 'w') as json_file:
        json.dump(lora_config_json, json_file)  # save to file

    # Matting
    if auto_matting:
        tmp_folder = f"{output_folder}_"
        os.rename(output_folder, tmp_folder)
        os.makedirs(output_folder, exist_ok=True)
        run_cmd = get_matting_cmd(tmp_folder, output_folder)
        run_cmd_with_log(run_cmd, api_call, log_file)

    return [
        output_folder,
        gr.update(value=f"`Images uploaded, {analysis_result}`"),
        gr.update(value=config_file_path),  # config_file_name
        lora_config_json,
        analysis_result
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
):
    if input_folder == '':
        return [
            "",
            gr.update(value=[], visible=False),  # hide Gallery
            gr.update(value=f"`Please upload images first!`"),
        ]

    if face_type == 'best_faces':
        output_folder = f"{input_folder}/../best_faces"
        face_type = 'whole_face'
        find_best = True
    else:
        output_folder = f"{input_folder}/../processed/{repeat}_{face_type}"
        find_best = False

    log_file = f"{lora_config_json['output_dir']}/log.txt"

    log.info(f'Processing images in {input_folder}...')

    run_cmd = f'accelerate launch "{os.path.join("DFL/mainscripts", "Extractor.py")}"'
    run_cmd += f' "{input_folder}"'
    run_cmd += f' "{output_folder}"'
    run_cmd += f' "{face_type}"'
    run_cmd += f' "{target_width}"'
    run_cmd += f' "{target_height}"'
    run_cmd += f' "100"'  # jpeg quality
    run_cmd += f' "{pass_ratio}"'

    # Deblur
    # trash_path = f"{input_folder}/../{repeat}_{face_type}_trash"
    # fixed_path = f"{input_folder}/../{repeat}_{face_type}_fix"
    # run_cmd3 = f'accelerate launch "{os.path.join("DeblurGANv2", "predict.py")}"'
    # run_cmd3 += f' "{trash_path}"'
    # run_cmd3 += f' "{fixed_path}"'
    # run_cmd3 += f' "{output_folder}"'

    def output():
        images = list(Path(output_folder).glob('*'))
        preview_images_dict.update({output_folder: images})
        return [
            f"{input_folder}/../processed",
            # show Gallery
            gr.update(value=[img for sublist in preview_images_dict.values() for img in sublist], visible=True),
            gr.update(value=f"`Face detection done, {face_type}`"),
        ]

    run_cmd_with_log(run_cmd, api_call, log_file)

    if find_best:
        # Sort
        run_cmd2 = f'accelerate launch "{os.path.join("DFL/mainscripts", "Sorter.py")}"'
        run_cmd2 += f' "{output_folder}"'
        run_cmd2 += f' "final"'  # Sort method
        run_cmd2 += f' "0"'  # Drop threshold
        run_cmd_with_log(run_cmd2, api_call, log_file)

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


def random_pick_image_and_flip(train_data_dir):
    flip_folder = '10_flip'
    tmp_dir = f'{train_data_dir}/../{flip_folder}'
    target_dir = f'{train_data_dir}/{flip_folder}'
    os.makedirs(tmp_dir, exist_ok=True)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        # os.makedirs(target_dir)

    for subdir, dirs, files in os.walk(train_data_dir):
        # skip 'head' dataset
        if 'head' in subdir:
            continue
        for file in random.sample(files, k=math.ceil(len(files) * 0.2)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(os.path.join(subdir, file), tmp_dir)

    for file_name in os.listdir(tmp_dir):
        image_path = os.path.join(tmp_dir, file_name)
        img = Image.open(image_path)
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_flipped.save(image_path)

    os.rename(tmp_dir, target_dir)


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
    run_cmd += f' --frequency_tags'  # save tags to tags.txt

    if not undesired_tags == '':
        run_cmd += f' --undesired_tags="{undesired_tags}"'
    run_cmd += f' "{train_data_dir}"'

    random_pick_image_and_flip(train_data_dir)

    run_cmd_with_log(run_cmd, api_call, f"{train_data_dir}/../output/log.txt")

    # Add prefix and postfix
    _add_pre_postfix(
        folder=train_data_dir,
        caption_file_ext='.txt',
        prefix=prefix,
        postfix=postfix,
        recursive=True
    )

    if os.path.exists(f"{train_data_dir}/../output/tags.txt"):
        with open(f"{train_data_dir}/../output/tags.txt", 'a') as f:
            f.write(f"prefix: {prefix}\n")
            f.write(f"postfix: {postfix}")

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
                visible=False
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
        )


def load_lora_config(use_wandb=True, mode='female', path='presets/lora/user_presets/lora_config.json'):
    if not os.path.exists(path):
        return {}

    with open(path) as f:
        config = json.load(f)

        seed = config['seed']
        if not seed:
            seed = 2076

        gender = '1girl'
        if mode == 'male':
            gender = '1boy'

        config['save_state'] = False
        config['save_every_n_steps'] = 0
        config['save_last_n_steps'] = 0
        config['save_last_n_steps_state'] = 0

        config['sample_every_n_epochs'] = 1
        config['sample_every_n_steps'] = 0
        config[
            'sample_prompts'] = f'(masterpiece, best quality), {gender}, solo, upper body, --n (worst quality, low quality:2), nudity, nsfw, cropped, lowres, watermark,  --w 512 --h 704 --l 7 --s 24  --d {seed}'
        config['sample_sampler'] = 'euler'

        if use_wandb:
            config['use_wandb'] = True
            config['wandb_api_key'] = '2b6dc47b76cf7235a360b3bcaa8ed1cd0e902969'
        else:
            config['use_wandb'] = False
            config['wandb_api_key'] = ''
        return config


def get_file_paths(directory):
    file_paths = []
    for file in os.listdir(directory):
        full_file_path = os.path.join(directory, file)
        if os.path.isfile(full_file_path):
            file_paths.append(full_file_path)
    return file_paths


def show_lora_files_(output_dir, file_postfix):
    files = get_file_paths(output_dir)
    filtered_files = []
    for f in files:
        if os.path.splitext(f)[1] == '.' + file_postfix:
            filtered_files.append(os.path.abspath(f))
    return gr.update(value=filtered_files, visible=True)


def show_lora_files(lora_config_json):
    return show_lora_files_(lora_config_json['output_dir'], lora_config_json['save_model_as'])


def check_lora_losses(lora_config_json):
    files = get_file_paths(lora_config_json['output_dir'])
    filtered_files = []
    for f in files:
        if os.path.splitext(f)[1] == '.' + lora_config_json['save_model_as']:
            filtered_files.append(os.path.abspath(f))

    with open('training_profile.json') as f:
        training_profile = json.load(f)
        retrain_range = training_profile.get('_retrain_range', [0.08, 0.09])

    max_epoch = lora_config_json['epoch']
    epoch_range = [max_epoch, max_epoch - 1, max_epoch - 2]
    need_retrain = True
    min_loss = 1
    max_loss = 0
    for file in filtered_files:
        basename = os.path.basename(file)
        match = re.match(r'last_epoch(\d+)_loss(0\.\d+)\.safetensors', basename)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            if epoch not in epoch_range:
                continue
            min_loss = min(min_loss, loss)
            max_loss = max(max_loss, loss)
            if retrain_range[0] <= loss <= retrain_range[1]:
                need_retrain = False

    log.info(f"Min loss of last 3 epoch {min_loss}")
    log.info(f"Max loss of last 3 epoch {max_loss}")
    if need_retrain:
        te_lr = lora_config_json['text_encoder_lr']
        unet_lr = lora_config_json['unet_lr']
        if max_loss < 0.08:
            log.info(f"Adjust learning rate [{te_lr}, {unet_lr}] -> [{te_lr * 0.5}, {unet_lr * 0.5}]")
            lora_config_json['text_encoder_lr'] = te_lr * 0.5
            lora_config_json['unet_lr'] = unet_lr * 0.5
        elif min_loss > 0.09:
            log.info(f"Adjust learning rate [{te_lr}, {unet_lr}] -> [{te_lr * 2}, {unet_lr * 2}]")
            lora_config_json['text_encoder_lr'] = te_lr * 2
            lora_config_json['unet_lr'] = unet_lr * 2
        return True, filtered_files

    return False, filtered_files


def _train_api(input_folder, model_path, trigger_words):
    uploaded_files = get_file_paths(input_folder)

    # Preprocess
    work_folder, _, _, config, face_stats = on_images_uploaded(uploaded_files, auto_matting=True, auto_upscale=True,
                                                               api_call=True)
    log.info(face_stats)
    if model_path:
        config.update({'pretrained_model_name_or_path': model_path})

    try:
        train_folder, _, _ = \
            process_images(work_folder, 'midfull_face_no_align', 512, 512, 10, 0, 1, {}, config, api_call=True)
        process_images(work_folder, 'whole_face_no_align', 512, 512, 20, 0, 1, {}, config, api_call=True)
        process_images(work_folder, 'head_no_align', 512, 768, 10, 0, 1, {}, config, api_call=True)

        # Caption
        caption_images(
            train_folder,
            0.5, 0.4,
            'SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
            'long hair, brown hair, black hair, brown eyes, black eyes, lips, teeth, nose, portrait, forehead',
            trigger_words,
            '',
            api_call=True
        )

        log.info(config)
        # Train
        config.pop('stop_text_encoder_training', None)
        config['stop_text_encoder_training_pct'] = 0  # Not yet supported
        train_model(headless=True, print_only=False, **config)

        retrain, files = check_lora_losses(config)
        if retrain:
            old_dir = os.path.join(config['output_dir'], 'old')
            os.makedirs(old_dir, exist_ok=True)
            for f in files:
                shutil.move(f, old_dir)
            train_model(headless=True, print_only=False, **config)

        return os.path.abspath(config['output_dir'])
    except Exception as e:
        stack = traceback.format_exc()
        print(stack)
        with open(f"{config['output_dir']}/log.txt", 'a') as f:
            f.write(stack)

        return os.path.abspath(config['output_dir'])


#----------------------------------------------------------------------
from retinaface.RetinaFace import build_model, detect_faces


face_model = build_model()
is_verifying = False


def _process_face_obj(obj):
    resp = []
    for face_idx in obj.keys():
        identity = obj[face_idx]
        facial_area = identity["facial_area"]
        y = facial_area[1]
        h = facial_area[3] - y
        x = facial_area[0]
        w = facial_area[2] - x
        img_region = [x, y, w, h]
        confidence = identity["score"]
        resp.append((img_region, confidence))
    return resp


async def _detect_api(input):
    from scheduler import download_image
    global is_verifying
    while is_verifying:
        log.info('Waiting other verify tasks to finish...')
        await asyncio.sleep(1)
    is_verifying = True
    try:
        with open('scheduler_settings/object_store.json') as f:
            obj_store_setting = json.load(f)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            download_folder = os.path.join('_workspace', 'verify', current_time)
            os.makedirs(download_folder, exist_ok=True)
            start_time = timeit.default_timer()
            if not download_image(input, download_folder, random.randint(0, 1000), obj_store_setting):
                return {
                    'valid': False,
                    'reason': 'Cannot fetch image',
                    'source': input
                }
            end_time = timeit.default_timer()
            log.info(f"Download finished in {(end_time - start_time)*1000:.2f} ms")

            start_time = timeit.default_timer()
            files = _get_image_file_paths(download_folder)
            detected_faces = _process_face_obj(detect_faces(files[0], 0.98, face_model))
            end_time = timeit.default_timer()
            log.info(f"Prediction finished in {(end_time - start_time)*1000:.2f} ms")

            print(detected_faces)

            if len(detected_faces) == 0:
                return {
                    'valid': False,
                    'reason': 'No human face',
                    'source': input
                }
            elif len(detected_faces) > 1:
                return {
                    'valid': False,
                    'reason': 'Multiple faces',
                    'source': input
                }
            else:
                fw = detected_faces[0][0][2]
                fh = detected_faces[0][0][3]
                if min(fw, fh) < 64:
                    return {
                        'valid': False,
                        'reason': 'Face too small',
                        'source': input
                    }
            return {
                'valid': True,
                'reason': '',
                'source': input
            }
    except Exception as e:
        return {
            'valid': False,
            'reason': str(e),
            'source': input
        }
    finally:
        is_verifying = False
#----------------------------------------------------------------------


def gradio_train_human_gui_tab(headless=False):
    with gr.Tab('人像训练'):
        upload_folder = gr.Textbox(visible=False)
        train_folder = gr.Textbox(visible=False)
        face_stats = gr.State({})
        preview_images_dict = gr.State({})

        info_text = gr.Markdown()

        with gr.Accordion('[Step 0] Select the image folder to upload'):
            with gr.Row(equal_height=False):
                upload_images = gr.File(
                    show_label=False,
                    file_count='directory',
                    # file_types=['image'],
                    sacle=2
                )
                with gr.Column(scale=0):
                    auto_matting = gr.Checkbox(value=True, label='Auto matting')
                    auto_upscale = gr.Checkbox(value=True, label='Auto upscale')
                    clear_upload_button = gr.Button('Clear uploaded images', variant='stop')
                    clear_train_button = gr.Button('Clear training folder', variant='stop')

        with gr.Row():
            images_preview = gr.Gallery(preview=True, columns=8, visible=False, scale=2)

        with gr.Accordion('[Step 1] Face detection and cropping', open=False):
            with gr.Row():
                face_type = gr.Dropdown(
                    label='Face detection type',
                    choices=[
                        'half_face',
                        'half_face_no_align',
                        'midfull_face',
                        'midfull_face_no_align',
                        'full_face',
                        'full_face_no_align',
                        'whole_face',
                        'whole_face_no_align',
                        'head',
                        'head_no_align',
                        'best_faces'
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
                lora_config_json = gr.JSON(load_lora_config, show_label=False, visible=False)

            (
                lora_train_data_dir,
                lora_reg_data_dir,
                lora_output_dir,
                lora_logging_dir,
                # new
                config_file_name,
            ) = lora_tab(headless=headless)

        with gr.Accordion('[Step 4] Results', open=False):
            show_files = gr.Button('Show trained LoRA files')
            lora_files = gr.File(
                interactive=False,
                visible=False
            )

        # Event listeners
        upload_images.upload(
            on_images_uploaded,
            inputs=[
                upload_images,  # files
                auto_matting,
                auto_upscale,
            ],
            outputs=[
                upload_folder,
                info_text,
                config_file_name,
                lora_config_json,
                face_stats
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
            ],
        )

        clear_upload_button.click(
            clear_upload_images,
            inputs=[upload_folder],
            outputs=[
                upload_folder,
                upload_images,
                info_text,
            ],
        )
        clear_train_button.click(
            clear_images,
            inputs=[train_folder, preview_images_dict],
            outputs=[
                images_preview,
                info_text,
            ],
        )

        show_files.click(
            show_lora_files,
            inputs=[lora_config_json],
            outputs=[lora_files]
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

        api_only_detect_input = gr.Textbox('', visible=False)
        api_only_detect_result = gr.JSON(visible=False)
        api_only_detect = gr.Button('', visible=False)
        api_only_detect.click(
            _detect_api,
            inputs=[api_only_detect_input],
            outputs=[api_only_detect_result],
            api_name='face_verify'
        )

    with gr.Tab('一键人像训练'):
        upload_folder = gr.Textbox(visible=False)
        output_folder = gr.Textbox(visible=False)
        lora_postfix = gr.Textbox('safetensors', visible=False)

        with gr.Accordion('Select the image folder to upload'):
            with gr.Row(equal_height=False):
                upload_images = gr.File(
                    show_label=False,
                    file_count='directory',
                    # file_types=['image'],
                    # sacle=2
                )
            model_path = gr.Textbox(label='底模路径（必填）', visible=False, value=None)  # 废弃
            trigger_words = gr.Textbox(label='触发词（必填）')
            train_button = gr.Button('开始训练', variant='primary')

        with gr.Accordion('Results', open=False):
            show_files = gr.Button('刷新LoRA文件')
            lora_files = gr.File(
                interactive=False,
                visible=False
            )

        upload_images.upload(
            on_images_uploaded_simple,
            inputs=[
                upload_images,  # files
            ],
            outputs=[
                upload_folder,
            ],
        )

        train_button.click(
            _train_api,
            inputs=[
                upload_folder,
                model_path,
                trigger_words,
            ],
            outputs=[
                output_folder
            ]
        )

        show_files.click(
            show_lora_files_,
            inputs=[output_folder, lora_postfix],
            outputs=[lora_files]
        )

    gradio_aurobit_upscale_gui_tab(headless=headless)

    gradio_aurobit_video_gui_tab(headless=headless)
