import datetime
import json
import os
import re
import requests
import shutil
import subprocess
import time
import traceback
import random
import math
import uuid
from PIL import Image
from pathlib import Path

import gradio as gr

from library.custom_logging import setup_logging
from library.aurobit_face_utils import verify_one2one

# Set up logging
log = setup_logging()


def detect_one2one(image_a, image_b):
    result = verify_one2one(image_a, image_b)
    return gr.update(value=result, visible=True)


def gradio_aurobit_face_verify_gui_tab(headless=False):
    with gr.Tab('人脸相似度检测(测试)'):
        with gr.Accordion('One to one'):
            with gr.Row():
                image_a = gr.Image(type='filepath')
                image_b = gr.Image(type='filepath')
            detect_one2one_btn = gr.Button(
                'Verify', variant='primary'
            )

            result_text = gr.Json(label='Result')

    detect_one2one_btn.click(
        detect_one2one,
        inputs=[
            image_a,
            image_b,
        ],
        outputs=[
            result_text
        ],
    )
