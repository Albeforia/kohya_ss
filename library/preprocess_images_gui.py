import datetime
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import gradio as gr

from library.custom_logging import setup_logging

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


def clear_images(input_folder):
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
    return [
        gr.update(value=[]),
        gr.update(value=f"`Data folder cleared`")
    ]


def process_images(
        input_folder,
        face_type,
        target_width,
        target_height,
        repeat,
        drop_threshold
        # progress=gr.Progress()
):
    if input_folder == '':
        return [
            "",
            gr.update(value=[]),
            gr.update(value=f"`Please upload images first!`")
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

    if os.name == 'posix':
        os.system(run_cmd)
        os.system(run_cmd2)
        return []  # TODO
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
        images = list(Path(output_folder).glob('*'))
        return [
            f"{input_folder}/../processed",
            gr.update(value=images),
            gr.update(value=f"`Face detection done, {face_type}`")
        ]


def gradio_preprocess_images_gui_tab(headless=False):
    with gr.Tab('数字分身'):
        upload_folder = gr.State('')
        train_folder = gr.State('')

        with gr.Accordion('Select the image folder to upload'):
            upload_images = gr.File(
                show_label=False,
                file_count='directory',
                # file_types=['image']
            )

        with gr.Accordion('Face detection and cropping', open=False):
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
                drop_threshold = gr.Slider(label='Discard blurry images', value=0.1, minimum=0, maximum=1,
                                           step=0.05)
            images_preview = gr.Gallery(preview=True, columns=8)

            with gr.Row():
                submit_images_button = gr.Button('Submit', variant='primary')
                clear_images_button = gr.Button('Clear', variant='stop')

        with gr.Accordion('Train'):
            pass

        with gr.Accordion('Info'):
            info_text = gr.Markdown()

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
            ],
            outputs=[
                train_folder,
                images_preview,
                info_text,
            ],
            show_progress=False,
            api_name='preprocess_images'
        )

        clear_images_button.click(
            clear_images,
            inputs=[train_folder],
            outputs=[
                images_preview,
                info_text,
            ],
            show_progress=False
        )
