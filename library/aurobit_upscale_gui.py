import os
import subprocess

import gradio as gr

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()


def submit_real_esrgan(image_path, scale_opt):
    output_folder = os.path.join('_workspace', 'upscale', 'esrgan')
    os.makedirs(output_folder, exist_ok=True)

    scale = 2
    if scale_opt == '4x':
        scale = 4
    elif scale_opt == '8x':
        scale = 8

    img_filename = os.path.basename(image_path)
    fname, fext = os.path.splitext(img_filename)
    sr_img_filename = f"{fname}_x{scale}{fext}"

    run_cmd = f'accelerate launch "{os.path.join("custom_scripts", "aurobit_upscale_script.py")}"'
    run_cmd += f' "--input_path={image_path}"'
    run_cmd += f' "--scale={scale}"'
    run_cmd += f' "--output_path={output_folder}"'

    p = subprocess.run(run_cmd, shell=True)
    if p.returncode == 0:
        return gr.update(value=os.path.join(output_folder, sr_img_filename))

    return gr.update(value=None)


def _upscale_api(image_path, scale_opt):
    output_folder = os.path.join('_workspace', 'upscale', 'esrgan')
    os.makedirs(output_folder, exist_ok=True)

    scale = 2
    if scale_opt == '4x':
        scale = 4
    elif scale_opt == '8x':
        scale = 8

    img_filename = os.path.basename(image_path)
    fname, fext = os.path.splitext(img_filename)
    sr_img_filename = f"{fname}_x{scale}{fext}"

    run_cmd = f'accelerate launch "{os.path.join("custom_scripts", "aurobit_upscale_script.py")}"'
    run_cmd += f' "--input_path={image_path}"'
    run_cmd += f' "--scale={scale}"'
    run_cmd += f' "--output_path={output_folder}"'

    p = subprocess.run(run_cmd, shell=True)
    if p.returncode == 0:
        return os.path.abspath(os.path.join(output_folder, sr_img_filename))

    return None


def gradio_aurobit_upscale_gui_tab(headless=False):
    with gr.Tab('超分辨率'):
        with gr.Accordion('Real-ESRGAN'):
            with gr.Row():
                image_a = gr.Image(type='filepath', label='input')
                image_b = gr.Image(type='filepath', interactive=False, label='output')

            with gr.Row():
                scale_opt0 = gr.Radio(['2x', '4x', '8x'], show_label=False, value='2x')
                submit_btn0 = gr.Button(
                    'Submit', variant='primary'
                )

        submit_btn0.click(
            submit_real_esrgan,
            inputs=[
                image_a,
                scale_opt0,
            ],
            outputs=[
                image_b
            ],
        )

        # API
        api_only_image_path = gr.Textbox('', visible=False)
        api_only_output_path = gr.Textbox('', visible=False)
        api_only_scale = gr.Textbox('', visible=False)
        api_only_upscale = gr.Button('', visible=False)
        api_only_upscale.click(
            _upscale_api,
            inputs=[api_only_image_path, api_only_scale],
            outputs=[api_only_output_path],
            api_name='upscale_esrgan'
        )
