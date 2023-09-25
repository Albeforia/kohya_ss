import datetime
import math
import os
import re
import subprocess
import uuid

import cv2
import gradio as gr
import imageio
import numpy as np
import requests
import yaml

from PIL import Image
from library.aurobit_fastblend_gui import FastModeRunner, AccurateModeRunner, InterpolationModeRunner
from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()


def load_sd_templates():
    l = sorted(os.listdir('custom_scripts/sd_templates/'))
    return [os.path.splitext(f)[0] for f in l]


def refresh_sd_templates():
    return gr.update(choices=load_sd_templates())


def images_to_video(img_folder, video_name, fps, sorter=None):
    from CoDeF.utils.video_visualizer import VideoVisualizer

    video_visualizer = VideoVisualizer(path=video_name, frame_size=None, fps=fps)

    images = [img for img in sorted(os.listdir(img_folder)) if
              img.endswith(".png") or img.endswith(".jpeg") or img.endswith(".jpg")]
    if sorter is not None:
        images = sorter(images)
    for img in images:
        print(img)
        image = cv2.imread(os.path.join(img_folder, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video_visualizer.add(image)
    video_visualizer.save()

    return True


def get_default_codef_config(width, height):
    config = {}
    config['img_wh'] = [width, height]
    config['canonical_wh'] = [width + 128, height + 128]
    config['lr'] = 0.001
    config['bg_loss'] = 0.003
    config['ref_idx'] = None
    config['N_xyz_w'] = [8]  # ?
    config['flow_loss'] = 0  # If have optical flow, should be 1
    config['flow_step'] = -1
    config['self_bg'] = True
    config['deform_hash'] = True
    config['vid_hash'] = True
    config['num_steps'] = 10000
    config['decay_step'] = [2500, 5000, 7500]
    config['annealed_begin_step'] = 4000
    config['annealed_step'] = 4000
    config['save_model_iters'] = 2000
    config['fps'] = 15

    return config


def _read_frames(frames_dir, sorter=None):
    file_names = sorted(os.listdir(frames_dir))
    if sorter is not None:
        file_names = sorter(file_names)
    video = []
    for file_name in file_names:
        file_path = os.path.join(frames_dir, file_name)
        image = imageio.v2.imread(file_path)
        image = np.array(image)
        video.append(image)
    return video


def _save_frames(frames, output_path):
    os.makedirs(output_path, exist_ok=True)
    for i, frame in enumerate(frames):
        frame = Image.fromarray(frame)
        frame.save(os.path.join(output_path, "%05d.jpg" % i))


def smooth_video_frames(
        video_guide,  # path
        video_style,  # path
        mode,
        window_size,
        batch_size,
        minimum_patch_size,
        num_iter,
        guide_weight,
        output_path,
        initialize,
        sorter,
        progress=gr.Progress(track_tqdm=True),
):
    # input
    frames_guide = _read_frames(video_guide, sorter)
    frames_style = _read_frames(video_style, sorter)
    # frames_guide, frames_style = align_frames(frames_guide, frames_style)

    frames_path = output_path
    os.makedirs(frames_path, exist_ok=True)
    # process
    ebsynth_config = {
        "minimum_patch_size": minimum_patch_size,
        "threads_per_block": 8,
        "num_iter": num_iter,
        "gpu_id": 0,
        "guide_weight": guide_weight,
        "initialize": initialize
    }
    if mode == "Fast":
        FastModeRunner().run(frames_guide, frames_style, batch_size=batch_size, window_size=window_size,
                             ebsynth_config=ebsynth_config, save_path=frames_path)
    elif mode == "Accurate":
        AccurateModeRunner().run(frames_guide, frames_style, batch_size=batch_size, window_size=window_size,
                                 ebsynth_config=ebsynth_config, save_path=frames_path)


def handle_video_upload(input_video, enable_codef):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('_workspace', 'video', current_time, 'sliced')
    os.makedirs(output_folder, exist_ok=True)

    def return_failure():
        return [
            gr.update(value=output_folder),
            gr.update(value='Error'),
            gr.update(value=None),
            gr.update(value=0), gr.update(value=0),
            gr.update(value=None),
            gr.update(value=None)
        ]

    target_size = 512

    run_cmd = f'accelerate launch "{os.path.join("custom_scripts", "aurobit_video_extract_script.py")}"'
    run_cmd += f' "--input_path={input_video}"'
    run_cmd += f' "--size={target_size}"'
    run_cmd += f' "--output_path={output_folder}"'

    p = subprocess.run(run_cmd, shell=True)
    if p.returncode != 0:
        return return_failure()

    # Now we have 3 folders: sliced, mask, matted

    canonical_image = None
    canonical_width = target_size
    canonical_height = target_size
    weight_file = None
    if enable_codef:
        config = get_default_codef_config(target_size, target_size)
        canonical_width = config['canonical_wh'][0]
        canonical_height = config['canonical_wh'][1]

        mask_dir = os.path.join(output_folder, '..', 'mask')
        codef_dir = os.path.join(output_folder, '..', 'CoDeF')
        model_dir = os.path.join(codef_dir, 'model')
        config_file = os.path.join(codef_dir, 'config.yaml')
        os.makedirs(codef_dir, exist_ok=True)
        with open(config_file, 'w') as file:
            yaml.dump(config, file)

        run_cmd2 = f'accelerate launch "{os.path.join("CoDeF", "train.py")}"'
        run_cmd2 += f' "--root_dir={output_folder}"'  # sliced frames
        run_cmd2 += f' "--model_save_path={model_dir}"'
        run_cmd2 += f' "--log_save_path={codef_dir}"'
        # run_cmd2 += f' "--mask_dir={mask_dir}"'
        run_cmd2 += f' "--encode_w"'
        run_cmd2 += f' "--annealed"'
        run_cmd2 += f' "--config={config_file}"'

        p = subprocess.run(run_cmd2, shell=True)
        if p.returncode != 0:
            return return_failure()

        # Delete unused weights
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file != "last.ckpt":
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                else:
                    weight_file = os.path.join(root, file)

        assert weight_file is not None
        run_cmd3 = f'accelerate launch "{os.path.join("CoDeF", "train.py")}"'
        run_cmd3 += f' "--test"'
        run_cmd3 += f' "--root_dir={output_folder}"'  # sliced frames
        run_cmd3 += f' "--weight_path={weight_file}"'
        run_cmd3 += f' "--log_save_path={codef_dir}"'
        # run_cmd3 += f' "--mask_dir={mask_dir}"'
        run_cmd3 += f' "--encode_w"'
        run_cmd3 += f' "--config={config_file}"'

        p = subprocess.run(run_cmd3, shell=True)
        if p.returncode != 0:
            return return_failure()
        canonical_image = os.path.join(output_folder, '..', 'results', 'exp', 'canonical_0.png')

    return [
        gr.update(value=output_folder),
        gr.update(value='Video slicing finished'),
        gr.update(value=canonical_image),  # preview
        gr.update(value=canonical_width), gr.update(value=canonical_height),
        gr.update(value=weight_file),
        gr.update(value=canonical_image)
    ]


def generate_images(source_folder, sd_address, sd_port, sd_template, sd_mode, img_prompt, width, height,
                    enable_codef, canonical_image, weight_file,
                    enable_temporalnet, temporal_weight):
    def sort(files):
        files.sort(key=lambda x: int(re.search(r'frame(\d+)', x).group(1)))
        return files

    if os.path.exists(source_folder) and os.path.isdir(source_folder) and os.listdir(source_folder):
        work_folder = os.path.join(source_folder, '..')
        generated_folder = os.path.join(work_folder, 'generated')
        os.makedirs(generated_folder, exist_ok=True)

        api_url = f'http://{sd_address}:{sd_port}'

        params_file = sd_template + '.json'

        run_cmd = f'accelerate launch "{os.path.join("custom_scripts", "aurobit_sd_script.py")}"'
        run_cmd += f' "--work_path={work_folder}"'
        run_cmd += f' "--api_addr={api_url}"'
        run_cmd += f' "--params_file={params_file}"'
        run_cmd += f' "--prompt={img_prompt}"'
        if sd_mode == 'none (txt2img)':
            run_cmd += f' "--mode=txt2img"'
        else:
            run_cmd += f' "--mode={sd_mode}"'
        run_cmd += f' "--w={width}"'
        run_cmd += f' "--h={height}"'
        if enable_codef:
            run_cmd += f' "--codef"'
            run_cmd += f' "--canonical_image={canonical_image}"'
        if enable_temporalnet:
            run_cmd += f' "--temporalnet"'
            run_cmd += f' "--temporal_weight={temporal_weight}"'

        log.info(run_cmd)
        p = subprocess.run(run_cmd, shell=True)
        if p.returncode == 0:
            canonical_trans_file = None
            out_video_path = None
            # Transform frames via CoDeF
            if enable_codef:
                canonical_trans_file = os.path.join(generated_folder, 'transformed.png')

                codef_dir = os.path.join(source_folder, '..', 'CoDeF')
                config_file = os.path.join(codef_dir, 'config.yaml')

                run_cmd3 = f'accelerate launch "{os.path.join("CoDeF", "train.py")}"'
                run_cmd3 += f' "--test"'
                run_cmd3 += f' "--root_dir={source_folder}"'  # sliced frames
                run_cmd3 += f' "--weight_path={weight_file}"'
                run_cmd3 += f' "--canonical_dir={generated_folder}"'
                run_cmd3 += f' "--log_save_path={codef_dir}"'
                # run_cmd3 += f' "--mask_dir={mask_dir}"'
                run_cmd3 += f' "--encode_w"'
                run_cmd3 += f' "--config={config_file}"'

                log.info(run_cmd3)
                p = subprocess.run(run_cmd3, shell=True)
                if p.returncode == 0:
                    out_video_path = os.path.join(work_folder, 'results', 'exp_transformed',
                                                  'video_exp_transformed.mp4')
            else:
                out_video_path = os.path.join(work_folder, f'{uuid.uuid4()}.mp4')
                if not images_to_video(generated_folder, out_video_path, 30, sorter=sort):
                    return [
                        gr.update(value=generated_folder),
                        gr.update(value=None),
                        gr.update(value="Video composition failed, try [Make GIF] below"),
                        gr.update(value=None),
                    ]

            return [
                gr.update(value=generated_folder),
                gr.update(value=out_video_path),
                gr.update(value='Generation finished'),
                gr.update(value=canonical_trans_file),
            ]

    else:
        return [
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value='Nothing to generate'),
            gr.update(value=None),
        ]


def smooth_video(source_folder, fb_mode, fb_window_size, fb_num_iter, fb_guide_weight):
    def sort(files):
        files.sort(key=lambda x: int(re.search(r'frame(\d+)', x).group(1)))
        return files

    work_folder = os.path.join(source_folder, '..')
    generated_folder = os.path.join(work_folder, 'generated')
    smoothed_path = os.path.join(work_folder, 'smooth')
    smooth_video_frames(
        video_guide=source_folder, video_style=generated_folder,
        mode=fb_mode,
        window_size=fb_window_size,
        batch_size=min(2 ** math.ceil(math.log2(2 * fb_window_size)), 128),
        minimum_patch_size=5,
        num_iter=fb_num_iter,
        guide_weight=fb_guide_weight,
        output_path=smoothed_path,
        initialize='identity',
        sorter=sort
    )
    out_video_path = os.path.join(work_folder, f'{uuid.uuid4()}_smooth.mp4')
    if not images_to_video(smoothed_path, out_video_path, 30):
        return [
            gr.update(value=None),
            gr.update(value="Video composition failed, try [Make GIF] below"),
        ]

    return [
        gr.update(value=out_video_path),
        gr.update(value='Generation finished'),
    ]


def list_models(sd_address, sd_port):
    api_url = f'http://{sd_address}:{sd_port}/sdapi/v1/sd-models'
    r = requests.get(url=api_url)
    if r.status_code == 200:
        models = r.json()
        return gr.update(value=models)

    return gr.update(value='Cannot connect to the server')


def list_cn(sd_address, sd_port):
    api_url = f'http://{sd_address}:{sd_port}/controlnet/control_types'
    r = requests.get(url=api_url)
    if r.status_code == 200:
        control_types = r.json()['control_types']
        return gr.update(value=control_types['All'])

    return gr.update(value='Cannot connect to the server')


def make_gif_fn(source_folder, final_size, final_duration, final_loop):
    if os.path.exists(source_folder) and os.path.isdir(source_folder) and os.listdir(source_folder):
        output_folder = os.path.join(source_folder, '..', 'output')
        os.makedirs(output_folder, exist_ok=True)

        run_cmd = f'accelerate launch "{os.path.join("custom_scripts", "aurobit_make_gif_script.py")}"'
        run_cmd += f' "--input_path={source_folder}"'
        run_cmd += f' "--size={final_size}"'
        run_cmd += f' "--duration={final_duration}"'
        run_cmd += f' "--loop={final_loop}"'
        run_cmd += f' "--output_path={output_folder}"'

        subprocess.run(run_cmd, shell=True)
        return [
            gr.update(value=output_folder),
            gr.update(value='GIF saved'),
        ]
    else:
        return [
            gr.update(value=None),
            gr.update(value='Invalid source folder'),
        ]


def gradio_aurobit_video_gui_tab(headless=False):
    with gr.Tab('Video (beta)'):
        source_folder = gr.Textbox(visible=False)
        generated_folder = gr.Textbox(visible=False)
        generated_width = gr.State(512)
        generated_height = gr.State(512)
        gif_folder = gr.Textbox(visible=False)
        codef_canonical_image = gr.Textbox(visible=False)
        codef_weight_path = gr.Textbox(visible=False)

        info_text = gr.Markdown()

        with gr.Accordion('上传视频 (< 3s)'):
            with gr.Row():
                input_video = gr.Video(show_label=False, scale=2)
                output_video = gr.PlayableVideo(show_label=False, scale=1)
            with gr.Row():
                enable_codef = gr.Checkbox(label='Enable CoDeF',
                                           info='When enabled, a CoDeF model will be automatically trained for the video (SLOW!!)',
                                           value=False, visible=False)
            with gr.Accordion('Canonical images', open=False, visible=False):
                with gr.Row():
                    canonical_image = gr.Image(label='Original', type='filepath')
                    canonical_trans = gr.Image(label='Transformed', type='filepath')

        with gr.Accordion('生成'):
            gr.Markdown(value='> Stable Diffusion WebUI should be launched with arguments: *--api --api-log*')
            with gr.Row():
                sd_address = gr.Textbox(label='Server IP for SD', value='127.0.0.1', scale=2)
                sd_port = gr.Textbox(label='Server port for SD', value='7860', scale=1)
            with gr.Row():
                with gr.Column():
                    sd_template = gr.Dropdown(
                        label='Templates',
                        choices=load_sd_templates(),
                        value='test',
                        scale=1
                    )
                    refresh_templates = gr.Button('Refresh')

                with gr.Column(scale=2):
                    sd_mode = gr.Radio(['none (txt2img)', 'source', 'matted', 'mask'], label='Initialization',
                                       value='none (txt2img)')
                    img_prompt = gr.Textbox(label='Prompt')

            with gr.Row():
                enable_temporalnet = gr.Checkbox(label='Use TemporalNet', value=False)
                temporal_weight = gr.Slider(label='TemporalNet weight', value=0.5, minimum=0, maximum=1, step=0.05)

            generate_btn = gr.Button('Generate', variant='primary')

            with gr.Row():
                fb_mode = gr.Radio(["Fast", "Accurate"], label="Inference mode", value="Accurate")
                fb_window_size = gr.Slider(label="Sliding window size", value=7, minimum=1, maximum=1000, step=1)
                fb_num_iter = gr.Slider(label="Number of iterations", value=5, minimum=1, maximum=10, step=1)
                fb_guide_weight = gr.Slider(label="Guide weight", value=10.0, minimum=0.0, maximum=100.0, step=0.1)

            smooth_btn = gr.Button('Smooth')

            with gr.Accordion('Info', open=False):
                with gr.Row(equal_height=False):
                    list_models_btn = gr.Button('List base models')
                    with gr.Accordion('Available models on the server', open=False):
                        models_info = gr.Json(show_label=False)
                with gr.Row(equal_height=False):
                    list_cn_btn = gr.Button('List ControlNets')
                    with gr.Accordion('Available ControlNets on the server', open=False):
                        cn_info = gr.Json(show_label=False)

        with gr.Accordion('生成 Gif', visible=False):
            with gr.Row():
                final_size = gr.Slider(label='Output size', value=256, minimum=128, maximum=512, step=64)
                final_duration = gr.Slider(label='Output duration', value=2, minimum=0.5, maximum=4, step=0.5)
                final_loop = gr.Checkbox(label='Loop', value=False, visible=False)
            make_gif = gr.Button('Make GIF', variant='primary')

        input_video.upload(
            handle_video_upload,
            inputs=[
                input_video,
                enable_codef
            ],
            outputs=[
                source_folder,
                info_text,
                canonical_image,
                generated_width, generated_height,
                codef_weight_path,
                codef_canonical_image,
            ]
        )

        generate_btn.click(
            generate_images,
            inputs=[
                source_folder,
                sd_address, sd_port,
                sd_template, sd_mode,
                img_prompt,
                generated_width, generated_height,
                enable_codef,
                codef_canonical_image,
                codef_weight_path,
                enable_temporalnet, temporal_weight,
            ],
            outputs=[
                generated_folder,
                output_video,
                info_text,
                canonical_trans
            ]
        )

        smooth_btn.click(
            smooth_video,
            inputs=[
                source_folder,
                fb_mode, fb_window_size, fb_num_iter, fb_guide_weight
            ],
            outputs=[
                output_video,
                info_text,
            ]
        )

        list_models_btn.click(
            list_models,
            inputs=[
                sd_address, sd_port
            ],
            outputs=[
                models_info
            ]
        )

        list_cn_btn.click(
            list_cn,
            inputs=[
                sd_address, sd_port
            ],
            outputs=[
                cn_info
            ]
        )

        make_gif.click(
            make_gif_fn,
            inputs=[
                generated_folder,
                final_size,
                final_duration,
                final_loop
            ],
            outputs=[
                gif_folder,
                info_text
            ]
        )

        refresh_templates.click(
            refresh_sd_templates,
            inputs=[],
            outputs=[sd_template]
        )
