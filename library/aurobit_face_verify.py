import datetime
import json
import math
import os
import random
import sys
import timeit

import cv2
import tensorflow as tf
from PIL import Image
from deepface.detectors import FaceDetector
from retinaface.RetinaFace import build_model, detect_faces

from library.aurobit_utils import download_image
from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

tf_init = False
is_verifying = False

import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=6)


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


def _convert_to_png(directory):
    for filename in os.listdir(directory):
        if not filename.endswith('.png'):
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


def _judge_faces(detected_faces, input_img, input_file):
    ratios = []
    for face in detected_faces:
        fw = face[0][2]
        fh = face[0][3]
        ratio = math.sqrt((fw * fh) / (input_img.shape[0] * input_img.shape[1]))
        ratios.append(ratio)

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


def _download_and_detect(input_file, obj_store_setting):
    global face_model
    global opencv_detector

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
    _convert_to_png(download_folder)
    files = _get_image_file_paths(download_folder)
    img = cv2.imread(files[0], cv2.IMREAD_COLOR)

    # First detect via OpenCV
    opencv_face_objs = FaceDetector.detect_faces(opencv_detector, 'opencv', img, False)
    opencv_results = []
    for _, img_region, confidence in opencv_face_objs:
        opencv_results.append((img_region, confidence))
    opencv_judge = _judge_faces(opencv_results, img, input_file)
    if not opencv_judge['valid']:
        # Detect using retinaface if OpenCV failed
        detected_faces = _process_face_obj(detect_faces(img, 0.9, face_model))
        end_time = timeit.default_timer()
        log.info(
            f"[retinaface] {input_file} {(end_time - start_time) * 1000:.2f} ms, {len(detected_faces)} faces")
        return _judge_faces(detected_faces, img, input_file)
    else:
        end_time = timeit.default_timer()
        log.info(
            f"[opencv] {input_file} {(end_time - start_time) * 1000:.2f} ms, {len(opencv_results)} faces")
        return opencv_judge


async def verify_face_api(input_file):
    global tf_init
    global face_model
    global opencv_detector
    global is_verifying

    if not tf_init:
        _init_tensorflow()
        tf_init = True

    if 'face_model' not in globals():
        face_model = build_model()
        opencv_detector = FaceDetector.build_model('opencv')
        # opencv_detector = {"face_detector": cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')}

    # while is_verifying:
    #     log.info('Waiting other verify tasks to finish...')
    #     await asyncio.sleep(1)
    is_verifying = True
    try:
        f = open('scheduler_settings/object_store.json')
        obj_store_setting = json.load(f)
        f.close()
        # return _download_and_detect(input_file, obj_store_setting)
        # result = await run_in_threadpool(_download_and_detect, input_file, obj_store_setting)
        result = await asyncio.get_event_loop().run_in_executor(executor, _download_and_detect, input_file,
                                                                obj_store_setting)
        return result

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
        print(_judge_faces(detected_faces, cv2.imread(file_path), file_path))
    else:
        pass
