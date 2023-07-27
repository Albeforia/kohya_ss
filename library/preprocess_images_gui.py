import os
import re
import subprocess
import time

import gradio as gr
from easygui import msgbox

from library.custom_logging import setup_logging
from .common_gui import get_folder_path

# Set up logging
log = setup_logging()

PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'


def process_images(
        input_folder,
        face_type,
        target_width,
        target_height,
        jpeg_quality,
        # progress=gr.Progress()
):
    if input_folder == '':
        msgbox('Input folder is missing...')
        return

    output_folder = f"{input_folder}/_processed"

    log.info(f'Processing images in {input_folder}...')
    # progress(0.0, desc="Start Processing")

    run_cmd = f'{PYTHON} "{os.path.join("DFL/mainscripts", "Extractor.py")}"'
    run_cmd += f' "{input_folder}"'
    run_cmd += f' "{output_folder}"'
    run_cmd += f' "{face_type}"'
    run_cmd += f' "{target_width}"'
    run_cmd += f' "{target_height}"'
    run_cmd += f' "{jpeg_quality}"'

    log.info(run_cmd)

    def check_progress():
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
        return output_folder
    else:
        subprocess.run(run_cmd)
        # process = subprocess.Popen(
        #     run_cmd,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE
        # )
        # for _ in progress.tqdm(check_progress(), desc="Processing"):
        #     pass
        return output_folder


def gradio_preprocess_images_gui_tab(headless=False):
    with gr.Tab('Preprocess Images'):
        with gr.Row():
            input_folder = gr.Textbox(
                label='Input folder',
                placeholder='Directory containing the images to group',
                interactive=True,
            )
            button_input_folder = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            button_input_folder.click(
                get_folder_path,
                outputs=input_folder,
                show_progress=False,
            )

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
            target_height = gr.Slider(label='Output height', value=768, minimum=256, maximum=2048, step=64)
            jpeg_quality = gr.Slider(label='Jpeg quality', value=90, minimum=1, maximum=100, step=5)

        # Output
        text_output = gr.Textbox(label='Output folder')

        submit_images_button = gr.Button('Submit')
        submit_images_button.click(
            process_images,
            inputs=[
                input_folder,
                face_type,
                target_width,
                target_height,
                jpeg_quality,
            ],
            outputs=[
                text_output,
            ],
            show_progress=False,
            api_name='preprocess_images'
        )
