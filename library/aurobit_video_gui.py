import datetime
import os
import subprocess
import uuid

import cv2
import gradio as gr
import requests

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()


def load_sd_templates():
    l = sorted(os.listdir('custom_scripts/sd_templates/'))
    return [os.path.splitext(f)[0] for f in l]


def refresh_sd_templates():
    return gr.update(choices=load_sd_templates())


def images_to_video(img_folder, video_name, fps):
    # 图片文件夹路径和视频文件名
    images = [img for img in os.listdir(img_folder) if
              img.endswith(".png") or img.endswith(".jpeg") or img.endswith(".jpg")]

    # 使用自定义的排序函数来处理帧序号
    # 假设所有的文件名都是以"frame"开头
    images.sort(key=lambda x: int(x[5:].split('.')[0]))

    # 读取第一张图片来获取尺寸信息
    frame = cv2.imread(os.path.join(img_folder, images[0]))
    height, width, layers = frame.shape

    # 创建视频编写器
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 将所有图片写入视频
    for image in images:
        video.write(cv2.imread(os.path.join(img_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def handle_video_upload(input_video):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('_workspace', 'video', current_time, 'sliced')
    os.makedirs(output_folder, exist_ok=True)

    run_cmd = f'accelerate launch "{os.path.join("custom_scripts", "aurobit_video_extract_script.py")}"'
    run_cmd += f' "--input_path={input_video}"'
    run_cmd += f' "--size=512"'
    run_cmd += f' "--output_path={output_folder}"'

    p = subprocess.run(run_cmd, shell=True)
    if p.returncode == 0:
        return [
            gr.update(value=output_folder),
            gr.update(value='Video slicing finished'),
        ]

    return [
        gr.update(value=output_folder),
        gr.update(value='Error'),
    ]


def generate_images(source_folder, sd_address, sd_port, sd_template, img_prompt):
    if os.path.exists(source_folder) and os.path.isdir(source_folder) and os.listdir(source_folder):
        work_folder = os.path.join(source_folder, '..')
        generated_folder = os.path.join(work_folder, 'generated')
        os.makedirs(generated_folder, exist_ok=True)

        api_url = f'http://{sd_address}:{sd_port}'

        params_file = sd_template + '.json'

        run_cmd = f'accelerate launch "{os.path.join("custom_scripts", "aurobit_sd_script.py")}"'
        run_cmd += f' "--work_path={work_folder}"'
        run_cmd += f' "--api_addr={api_url}"'
        run_cmd += f' "--mode=txt2img"'  # TODO
        run_cmd += f' "--params_file={params_file}"'
        run_cmd += f' "--prompt={img_prompt}"'
        run_cmd += f' "--output_path={generated_folder}"'

        p = subprocess.run(run_cmd, shell=True)
        if p.returncode == 0:
            out_video_path = os.path.join(work_folder, f'{uuid.uuid4()}.mp4')
            images_to_video(generated_folder, out_video_path, 15)
            return [
                gr.update(value=generated_folder),
                gr.update(value=out_video_path),
                gr.update(value='Generation finished'),
            ]
        return [
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value='Error'),
        ]
    else:
        return [
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value='Nothing to generate'),
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
    with gr.Tab('Video'):
        source_folder = gr.Textbox(visible=False)
        generated_folder = gr.Textbox(visible=False)
        gif_folder = gr.Textbox(visible=False)

        info_text = gr.Markdown()

        with gr.Accordion('上传视频'):
            with gr.Row():
                input_video = gr.Video(show_label=False)
                output_video = gr.Video(show_label=False, interactive=False)

        with gr.Accordion('生成'):
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

                img_prompt = gr.Textbox(label='Prompt', scale=2)

            generate_btn = gr.Button('Generate', variant='primary')
            with gr.Row(equal_height=False):
                list_models_btn = gr.Button('List base models')
                with gr.Accordion('Available models on the server', open=False):
                    models_info = gr.Json(show_label=False)
            with gr.Row(equal_height=False):
                list_cn_btn = gr.Button('List ControlNets')
                with gr.Accordion('Available ControlNets on the server', open=False):
                    cn_info = gr.Json(show_label=False)

        # with gr.Accordion('[Step 2] 输出Gif'):
        #     with gr.Row():
        #         final_size = gr.Slider(label='Output size', value=256, minimum=128, maximum=512, step=64)
        #         final_duration = gr.Slider(label='Output duration', value=2, minimum=0.5, maximum=4, step=0.5)
        #         final_loop = gr.Checkbox(label='Loop', value=False)
        #
        #     make_gif = gr.Button('Make GIF', variant='primary')

        input_video.upload(
            handle_video_upload,
            inputs=[
                input_video
            ],
            outputs=[
                source_folder,
                info_text
            ]
        )

        generate_btn.click(
            generate_images,
            inputs=[
                source_folder,
                sd_address, sd_port,
                sd_template,
                img_prompt,
            ],
            outputs=[
                generated_folder,
                output_video,
                info_text
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

        # make_gif.click(
        #     make_gif_fn,
        #     inputs=[
        #         generated_folder,
        #         final_size,
        #         final_duration,
        #         final_loop
        #     ],
        #     outputs=[
        #         gif_folder,
        #         info_text
        #     ]
        # )

        refresh_templates.click(
            refresh_sd_templates,
            inputs=[],
            outputs=[sd_template]
        )
