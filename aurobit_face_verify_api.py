import argparse
import os

import gradio as gr

from library.aurobit_face_verify import verify_face_api
from library.aurobit_utils import check_necessary_files
from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()


def UI(**kwargs):
    check_necessary_files(scheduler_only=True)

    css = ''

    headless = kwargs.get('headless', True)
    log.info(f'headless: {headless}')

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            log.info('Load CSS...')
            css += file.read() + '\n'

    if os.path.exists('./.release'):
        with open(os.path.join('./.release'), 'r', encoding='utf8') as file:
            release = file.read()

    if os.path.exists('./README.md'):
        with open(os.path.join('./README.md'), 'r', encoding='utf8') as file:
            README = file.read()

    interface = gr.Blocks(
        css=css, title=f'Kohya_ss GUI {release}', theme=gr.themes.Default()
    )

    with interface:
        with gr.Accordion('Face Verify'):
            api_only_detect_input = gr.Textbox('', visible=False)
            api_only_detect_result = gr.JSON(visible=False)
            api_only_detect = gr.Button('', visible=False)
            api_only_detect.click(
                verify_face_api,
                inputs=[api_only_detect_input],
                outputs=[api_only_detect_result],
                api_name='face_verify'
            )

    # Show the interface
    launch_kwargs = {}
    username = kwargs.get('username')
    password = kwargs.get('password')
    server_port = kwargs.get('server_port', 0)
    inbrowser = kwargs.get('inbrowser', False)
    share = kwargs.get('share', False)
    server_name = kwargs.get('listen')

    launch_kwargs['server_name'] = server_name
    if username and password:
        launch_kwargs['auth'] = (username, password)
    if server_port > 0:
        launch_kwargs['server_port'] = server_port
    if inbrowser:
        launch_kwargs['inbrowser'] = inbrowser
    if share:
        launch_kwargs['share'] = share

    # FIXME When queue is enabled, we have to turn off the network proxy otherwise everything is irresponsible
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/9074
    interface.launch(**launch_kwargs)


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )
    parser.add_argument(
        '--headless', action='store_true', help='Is the server headless'
    )

    args = parser.parse_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
        headless=args.headless,
    )
