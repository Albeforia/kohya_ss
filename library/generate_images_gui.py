import datetime
import json
import os
import re
import shutil
import subprocess
import time
import traceback
import random
import math
from PIL import Image
from pathlib import Path

import gradio as gr

import library.train_util as train_util
from library.custom_logging import setup_logging
from lora_gui import lora_tab, train_model

# Set up logging
log = setup_logging()

def gradio_generate_images_gui(headless=False):
    with gr.Accordion('Generate images', open=False):
        with gr.Row():
            server = gr.Textbox('127.0.0.1:7860', label='Server address for SD')
            #
