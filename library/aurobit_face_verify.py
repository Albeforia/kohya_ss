import asyncio
import datetime
import json
import math
import os
import random
import sys
import timeit
from datetime import datetime

import requests
import tensorflow as tf
from PIL import Image
from minio import Minio
from minio.error import S3Error
from retinaface.RetinaFace import build_model, detect_faces

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

tf_init = False
is_verifying = False


def create_minio_client(setting):
    provider = setting['provider']
    secret_id = setting['secret_id']
    secret_key = setting['secret_key']
    region = setting['region']
    if provider == 'tencent':
        # endpoint = format_endpoint(None, region, u'cos.', True, True)
        pass
    elif provider == 'amazon':
        endpoint = f"s3.{region}.amazonaws.com"
    else:
        return None
    return Minio(
        endpoint,
        access_key=secret_id,
        secret_key=secret_key,
        secure=True
    )


def download_image(url, path, index, setting):
    def convert_bytes(size, unit=None):
        if unit == "KB":
            return print('File size: ' + str(round(size / 1024, 2)) + ' Kilobytes')
        elif unit == "MB":
            return print('File size: ' + str(round(size / (1024 * 1024), 2)) + ' Megabytes')
        elif unit == "GB":
            return print('File size: ' + str(round(size / (1024 * 1024 * 1024), 2)) + ' Gigabytes')
        else:
            return print('File size: ' + str(size) + ' bytes')

    client = create_minio_client(setting)

    use_cdn = setting.get('use_cdn', False)
    cdn = setting.get('cdn', '')

    try:
        print(f'Downloading {url}')
        _, ext = os.path.splitext(url)
        local_file = os.path.join(path, f'image_{index}{ext}')
        if not use_cdn:
            # Directly download from S3
            client.fget_object(
                bucket_name=setting['bucket'],
                object_name=url,
                file_path=local_file)
        else:
            response = requests.get(cdn + '/' + url)
            response.raise_for_status()
            with open(local_file, 'wb') as f:
                f.write(response.content)
        convert_bytes(os.path.getsize(local_file), 'MB')
        return True
    except (S3Error, requests.HTTPError) as e:
        print(f"[download_image] Network error: {e}")
        return False


def _get_image_file_paths(directory):
    image_file_paths = []
    for file in os.listdir(directory):
        full_file_path = os.path.join(directory, file)
        if os.path.isfile(full_file_path):
            # Check the file extension
            _, ext = os.path.splitext(full_file_path)
            if ext.lower() in ['.jpg', '.jpeg', '.png']:
                image_file_paths.append(full_file_path)
    return image_file_paths


def _convert_webp_to_png(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.webp'):
            img = Image.open(os.path.join(directory, filename))
            png_filename = filename.rstrip('.webp') + '.png'
            img.save(os.path.join(directory, png_filename), 'PNG')
            os.remove(os.path.join(directory, filename))


def _init_tensorflow():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6500)])
        except RuntimeError as e:
            log.error(e)


def _process_face_obj(obj):
    resp = []
    if not isinstance(obj, dict):
        return resp
    for face_idx in obj.keys():
        identity = obj[face_idx]
        facial_area = identity["facial_area"]
        y = facial_area[1]
        h = facial_area[3] - y
        x = facial_area[0]
        w = facial_area[2] - x
        img_region = [x, y, w, h]
        confidence = identity["score"]
        resp.append((img_region, confidence))
    return resp


def _judge_faces(detected_faces, local_file, input_file):
    img = Image.open(local_file)

    ratios = []
    for face in detected_faces:
        fw = face[0][2]
        fh = face[0][3]
        ratio = math.sqrt((fw * fh) / (img.size[0] * img.size[1]))
        ratios.append(ratio)

    print(ratios)

    if len(detected_faces) == 0:
        return {
            'valid': False,
            'reason': 'No human face',
            'source': input_file
        }
    elif len(detected_faces) > 1:
        max2 = sorted(ratios)[-2:]
        if (max2[1] / max2[0]) < 1.8:
            return {
                'valid': False,
                'reason': 'Multiple faces',
                'source': input_file
            }
    else:
        # Only one face
        if ratios[0] < 0.06:
            return {
                'valid': False,
                'reason': 'Face too small',
                'source': input_file
            }
    return {
        'valid': True,
        'reason': '',
        'source': input_file
    }


async def verify_face_api(input_file):
    global tf_init
    global face_model
    global is_verifying

    if not tf_init:
        _init_tensorflow()
        tf_init = True

    if 'face_model' not in globals():
        face_model = build_model()

    while is_verifying:
        log.info('Waiting other verify tasks to finish...')
        await asyncio.sleep(1)
    is_verifying = True
    try:
        with open('scheduler_settings/object_store.json') as f:
            obj_store_setting = json.load(f)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + str(random.randint(0, 1000))
            download_folder = os.path.join('_workspace', 'verify', current_time)
            os.makedirs(download_folder, exist_ok=True)
            start_time = timeit.default_timer()
            if not download_image(input_file, download_folder, 0, obj_store_setting):
                return {
                    'valid': False,
                    'reason': 'Cannot fetch image',
                    'source': input_file
                }
            end_time = timeit.default_timer()
            log.info(f"Download finished in {(end_time - start_time) * 1000:.2f} ms")

            start_time = timeit.default_timer()
            _convert_webp_to_png(download_folder)
            files = _get_image_file_paths(download_folder)
            detected_faces = _process_face_obj(detect_faces(files[0], 0.9, face_model))
            end_time = timeit.default_timer()
            log.info(f"Prediction finished in {(end_time - start_time) * 1000:.2f} ms")

            return _judge_faces(detected_faces, files[0], input_file)

    except Exception as e:
        return {
            'valid': False,
            'reason': str(e),
            'source': input_file
        }
    finally:
        is_verifying = False


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
        detected_faces = _process_face_obj(detect_faces(file_path, 0.9, build_model()))
        print(_judge_faces(detected_faces, file_path, file_path))
    else:
        pass
