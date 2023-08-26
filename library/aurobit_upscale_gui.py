import datetime
import os
import subprocess

import gradio as gr

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()


def upscale_real_esrgan(image_path, scale_opt):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('_workspace', 'upscale', 'esrgan', current_time)
    os.makedirs(output_folder, exist_ok=True)

    scale = 2
    if scale_opt == '4x':
        scale = 4
    elif scale_opt == '8x':
        scale = 8

    run_cmd = f'accelerate launch "{os.path.join("custom_scripts", "aurobit_upscale_script.py")}"'
    run_cmd += f' "--input_path={image_path}"'
    run_cmd += f' "--scale={scale}"'
    run_cmd += f' "--output_path={output_folder}"'

    p = subprocess.run(run_cmd, shell=True)
    if p.returncode == 0:
        results = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]
        if len(results) == 1:  # Single image
            return os.path.join(output_folder, results[0])
        else:
            return output_folder

    return None


def submit_real_esrgan(image_path, scale_opt):
    return gr.update(value=upscale_real_esrgan(image_path, scale_opt))


def upscale_codeformer(image_path,
                       cf_weight,
                       cf_face_aligned,
                       cf_face_upscale,
                       cf_bg_upscale):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('_workspace', 'upscale', 'codeformer', current_time)
    os.makedirs(output_folder, exist_ok=True)

    final_scale = 1
    if cf_face_upscale or cf_bg_upscale:
        final_scale = 2

    run_cmd = f'accelerate launch "{os.path.join("CodeFormer", "inference_codeformer.py")}"'
    run_cmd += f' "--input_path={image_path}"'
    run_cmd += f' "--output_path={output_folder}"'
    run_cmd += f' "--fidelity_weight={cf_weight}"'
    run_cmd += f' "--upscale={final_scale}"'
    if cf_face_aligned:
        run_cmd += f' "--has_aligned"'
    if cf_bg_upscale:
        run_cmd += f' "--bg_upsampler=realesrgan"'
    if cf_face_upscale:
        run_cmd += f' "--face_upsample"'

    p = subprocess.run(run_cmd, shell=True)
    if p.returncode == 0:
        final_folder = 'final_results' if not cf_face_aligned else 'restored_faces'
        final_folder = os.path.abspath(os.path.join(output_folder, final_folder))
        results = [f for f in os.listdir(final_folder) if os.path.isfile(os.path.join(final_folder, f))]
        if len(results) == 1:  # Single image
            return os.path.join(final_folder, results[0])
        else:
            return final_folder

    return None


def submit_codeformer(image_path,
                      cf_weight,
                      cf_face_aligned,
                      cf_face_upscale,
                      cf_bg_upscale):
    return gr.update(value=upscale_codeformer(image_path, cf_weight, cf_face_aligned, cf_face_upscale, cf_bg_upscale))


def _upscale_api(image_path, method):
    method_int = int(method)
    if method_int == 0:
        return upscale_real_esrgan(image_path, '2x')
    elif method_int == 1:
        return upscale_codeformer(image_path, 0.85, False, True, True)
    else:
        return upscale_real_esrgan(image_path, '2x')


def gradio_aurobit_upscale_gui_tab(headless=False):
    with gr.Tab('高清化'):
        with gr.Accordion('Real-ESRGAN'):
            with gr.Row():
                image_esrgan = gr.Image(type='filepath', label='input')
                image_esrgan_out = gr.Image(type='filepath', interactive=False, label='output')

            with gr.Row():
                scale_opt0 = gr.Radio(['2x', '4x', '8x'], show_label=False, value='2x')
                submit_btn0 = gr.Button(
                    'Submit', variant='primary'
                )

        submit_btn0.click(
            submit_real_esrgan,
            inputs=[
                image_esrgan,
                scale_opt0,
            ],
            outputs=[
                image_esrgan_out
            ],
        )

        with gr.Accordion('CodeFormer'):
            with gr.Row():
                image_cf = gr.Image(type='filepath', label='input')
                image_cf_out = gr.Image(type='filepath', interactive=False, label='output')

            with gr.Row():
                cf_weight = gr.Slider(label='Fidelity weight', value=0.7, minimum=0, maximum=1, step=0.05,
                                      info='Balance the quality and fidelity')
                cf_face_aligned = gr.Checkbox(label='Aligned', info='Input are cropped and aligned faces')
                cf_face_upscale = gr.Checkbox(label='Upscale face')
                cf_bg_upscale = gr.Checkbox(label='Upscale background')
            with gr.Row():
                submit_btn1 = gr.Button(
                    'Submit', variant='primary'
                )

        submit_btn0.click(
            submit_real_esrgan,
            inputs=[
                image_esrgan,
                scale_opt0,
            ],
            outputs=[
                image_esrgan_out
            ],
        )

        submit_btn1.click(
            submit_codeformer,
            inputs=[
                image_cf,
                cf_weight,
                cf_face_aligned,
                cf_face_upscale,
                cf_bg_upscale,
            ],
            outputs=[
                image_cf_out
            ],
        )

        # API
        api_only_image_path = gr.Textbox('', visible=False)
        api_only_output_path = gr.Textbox('', visible=False)
        api_only_method = gr.Textbox('', visible=False)
        api_only_upscale = gr.Button('', visible=False)
        api_only_upscale.click(
            _upscale_api,
            inputs=[api_only_image_path, api_only_method],
            outputs=[api_only_output_path],
            api_name='upscale'
        )
