import datetime
import os
import subprocess

import gradio as gr

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()


def face_fusion(image_template, image_target):
    import cv2
    from PIL import Image
    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    image_face_fusion = pipeline(Tasks.image_face_fusion,
                                 model='damo/cv_unet-image-face-fusion_damo')

    template_path = image_template
    user_path = image_target
    result = image_face_fusion(dict(template=template_path, user=user_path))
    rgb_image = cv2.cvtColor(result[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
    return gr.update(value=Image.fromarray(rgb_image))


def gradio_aurobit_face_fusion_gui_tab(headless=False):
    with gr.Tab('Face Fusion'):
        with gr.Row():
            image_template = gr.Image(type='filepath', label='Template')
            image_target = gr.Image(type='filepath', label='User')

        with gr.Row(equal_height=False):
            submit_btn0 = gr.Button('Submit', variant='primary')
            image_result = gr.Image(show_label=False, type='pil')

    submit_btn0.click(
        face_fusion,
        inputs=[
            image_template,
            image_target,
        ],
        outputs=[
            image_result
        ],
    )
