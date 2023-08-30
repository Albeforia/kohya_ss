import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import time
import traceback
from datetime import datetime

import requests
import schedule
from PIL import Image
from confluent_kafka import Consumer
from gradio_client import Client
from pymongo import MongoClient, ReturnDocument
from qcloud_cos import CosClientError, CosServiceError
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client


def get_last_line(file_name):
    with open(file_name, 'rb') as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()
    return last_line


def convert_webp_to_png(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.webp'):
            img = Image.open(os.path.join(directory, filename))
            png_filename = filename.rstrip('.webp') + '.png'
            img.save(os.path.join(directory, png_filename), 'PNG')
            os.remove(os.path.join(directory, filename))


def create_cos_client(setting):
    secret_id = setting['secret_id']
    secret_key = setting['secret_key']
    region = setting['region']
    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=None, Scheme='https')
    return CosS3Client(config)


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

    client = create_cos_client(setting)

    try:
        print(f'Downloading {url}')
        _, ext = os.path.splitext(url)
        local_file = os.path.join(path, f'image_{index}{ext}')
        response = client.download_file(
            Bucket=setting['bucket'],
            Key=url,
            DestFilePath=local_file)
        convert_bytes(os.path.getsize(local_file), 'MB')
    except CosClientError or CosServiceError as e:
        print(f"[download_image] Network error: {e}")
        return False


def lora_task(task_id, user_id, task_params, setting):
    def failure(reason):
        return {
            'start_time': start_time,
            'finish_time': datetime.now(),
            'status': 4,
            'info': f'Task fail: {reason}'
        }

    start_time = datetime.now()
    # 创建以 task_id 命名的目录
    img_dir = os.path.join('tmp', task_id + '_images')
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir, exist_ok=True)

    try:
        # 从params中获取img字段，然后下载所有的图片
        img_url = task_params.get('frontImg')
        if img_url is not None:
            if not download_image(img_url, img_dir, 0, setting['obj_store']):
                return failure('Cannot fetch images')
        img_urls = task_params.get('otherImg', [])
        for i, url in enumerate(img_urls):
            if not download_image(url, img_dir, i + 1, setting['obj_store']):
                return failure('Cannot fetch images')

        convert_webp_to_png(img_dir)

        num_files = len(os.listdir(img_dir))
        print(f'[{task_id}] Image downloading finished, total {num_files}')
        if num_files <= 0:
            return failure('No images')

        # Call training API
        print("Start training")
        client = Client(setting['api_addr'])

        a = os.path.abspath(img_dir)
        c = user_id[-8:] if len(user_id) >= 8 else user_id
        result = client.predict(
            a, '', c, api_name="/train_lora",
        )
        if result is not None:
            log_file = os.path.join(result, 'log.txt')
            last_line = get_last_line(log_file)
            if last_line == 'SUCCESS':
                return {
                    'start_time': start_time,
                    'finish_time': datetime.now(),
                    'status': 3,
                    'info': "Task success: Training finished",
                    'files': result
                }
            else:
                return failure(f"Error occurred when training: {last_line} \n Refer to the log file {log_file}")
    except Exception as e:
        traceback.print_stack()
        return failure(str(e))


def highres_task(task_id, user_id, task_params, setting):
    def failure(reason):
        return {
            'start_time': start_time,
            'finish_time': datetime.now(),
            'status': 4,
            'info': f'Task highres fail: {reason}'
        }

    start_time = datetime.now()
    # 创建以 task_id 命名的目录
    img_dir = os.path.join('tmp', task_id + '_images')
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir, exist_ok=True)

    try:
        # 从params中获取img字段，然后下载所有的图片
        img_url = task_params.get('img')
        if img_url is not None:
            if not download_image(img_url, img_dir):
                return failure('Cannot fetch images')

        files = os.listdir(img_dir)
        num_files = len(files)
        print(f'[{task_id}] Image downloading finished, total {num_files}')
        if num_files <= 0:
            return failure('No images')

        # Call training API
        print("Start up-scaling")
        client = Client(setting['api_addr'])

        for f in files:
            result = client.predict(
                os.path.abspath(os.path.join(img_dir, f)), '0', api_name="/upscale",
            )
            if result is not None:
                if os.path.exists(result):
                    return {
                        'start_time': start_time,
                        'finish_time': datetime.now(),
                        'status': 3,
                        'info': "Task success: Up-scaling finished",
                        'files': result
                    }
                else:
                    return failure(f"Error occurred when up-scaling")
    except Exception as e:
        traceback.print_stack()
        return failure(str(e))


def upload_trained_files(output_path, user_id, task_id, setting):
    client = create_cos_client(setting)

    loss_ranges = [[0.08, 0.09], [0.07, 0.115], [0.06, 0.125], [0, 1]]
    files = glob.glob(os.path.join(output_path, '*.safetensors'))
    pattern = r'last_epoch(\d+)_loss(0\.\d+)\.safetensors'

    # 寻找最大的 epoch
    max_epoch = 0
    for file in files:
        match = re.match(pattern, os.path.basename(file))
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch

    # 上传文件
    uploaded = []
    uploaded_local = []
    for loss_range in loss_ranges:
        if len(uploaded) >= 4:
            break
        for file in files:
            if file in uploaded_local:
                continue
            basename = os.path.basename(file)
            match = re.match(pattern, basename)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                if epoch > max_epoch * 0.5 and loss_range[0] <= loss <= loss_range[1]:
                    print(f'Uploading file: {file}')
                    with open(file, 'rb') as fp:
                        fname = f"duck-test/{user_id}/{task_id}/{task_id}_{basename}"
                        try:
                            response = client.put_object(
                                Bucket=setting['bucket'],
                                Body=fp,
                                Key=fname,
                            )
                            uploaded.append(fname)
                            uploaded_local.append(file)
                        except (CosClientError, CosServiceError) as e:
                            print(f"Upload failed, {e.get_error_code()} {e.get_error_msg()}")

    return uploaded


def upload_single_image(image_path, key, setting):
    client = create_cos_client(setting)

    # 上传文件
    print(f'Uploading file: {image_path}')
    with open(image_path, 'rb') as fp:
        try:
            response = client.put_object(
                Bucket=setting['bucket'],
                Body=fp,
                Key=key,
            )
        except (CosClientError, CosServiceError) as e:
            print(f"Upload failed, {e.get_error_code()} {e.get_error_msg()}")


def handle_lora_result(result, user_id, task_id, collection_result, webhook, setting):
    file_list = upload_trained_files(result['files'], user_id, task_id, setting['obj_store'])
    collection_result.update_one(
        {'userId': user_id},
        {'$set': {
            'result': file_list,
            'taskId': task_id,
            'updateTime': datetime.now(),
        }},
        upsert=True
    )
    data = {
        'userId': user_id,
        'taskId': task_id,
        'finishTime': result['finish_time'].strftime("%Y-%m-%d %H:%M:%S"),
        "taskTpye": "test",
        "name": "test",
        'status': True,
        'reason': ''
    }
    data = json.dumps(data, indent=4, default=str)
    headers = {
        'Content-Type': 'application/json',
    }
    response = requests.request("POST", webhook, data=data, headers=headers)
    print(f"webhook: {response}")


def handle_highres_result(result, user_id, task_id, task_params, collection_result, webhook, setting):
    def add_watermark(input_path, output_path):
        run_cmd = f'accelerate launch "{os.path.join("watermark", "watermark.py")}"'
        run_cmd += f' "--input_path={input_path}"'
        run_cmd += f' "--output_path={output_path}"'
        subprocess.run(run_cmd, shell=True)
        filename, ext = os.path.splitext(input_path)
        return filename + '_w' + ext

    headers = {
        'Content-Type': 'application/json',
    }

    doc = collection_result.find_one({'taskId': task_params['relateId']})
    if doc is None or doc['userId'] != user_id:
        print('Up-scaling is done, but cannot find corresponding result')
        return

    target_img = task_params['img']
    result_imgs = doc['result']
    result_imgs_watermark = doc['resultWithWaterMark']
    index = next((i for i, x in enumerate(result_imgs) if x == target_img), None)
    if index is not None:
        highres_img = os.path.abspath(result['files'])
        highres_img_w = add_watermark(highres_img, os.path.dirname(highres_img))
        print(f"Found target image, will replace with a up-scaled version")
        upload_single_image(highres_img, target_img, setting['obj_store'])
        upload_single_image(highres_img_w, result_imgs_watermark[index]['img'], setting['obj_store'])
        result_imgs_watermark[index]['hdFlag'] = True
    else:
        print('Cannot find the image to up-scale!')

    # Write back to DB
    collection_result.update_one(
        {'taskId': task_params['relateId']},
        {'$set': {
            'resultWithWaterMark': result_imgs_watermark,
            'updateTime': datetime.now(),
        }}
    )

    data = {
        'userId': user_id,
        'img': result_imgs_watermark[index]['img'],
        'relateId': task_params['relateId']
    }
    data = json.dumps(data, indent=4, default=str)
    response = requests.request("POST", webhook, data=data, headers=headers)
    print(f"webhook: {response}")


def scan_and_do_task(consumer, collection, collection_result, setting):
    # print('Start comsuming...')
    while True:
        msg = consumer.poll(0.1)  # 0.1为超时时间
        if msg is None:
            # print('No message')
            break
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        task_id = msg.value().decode('utf-8')
        print(f"Received message: {task_id}")
        # 更新MongoDB中的文档状态
        # TODO Should set status after task_type is filtered?
        updated_doc = collection.find_one_and_update(
            {'taskId': task_id},
            {'$set': {'status': 2}},  # 2=Task is executing
            return_document=ReturnDocument.AFTER
        )
        if updated_doc is not None:
            user_id = updated_doc.get('userId')
            task_type = updated_doc.get('taskType')
            task_params = updated_doc.get('params', {})

            retry_times = 0
            result = None
            while retry_times < 3:
                if task_type == 'lora_train':
                    result = lora_task(task_id, user_id, task_params, setting)
                elif task_type == 'highres_task':
                    # highres_task uses different params
                    task_params = updated_doc.get('taskInfo', {})
                    result = highres_task(task_id, user_id, task_params, setting)
                else:
                    print(f"Unknown task type {task_type}, pass")
                    break

                if result['status'] == 4:
                    print(f"{result['info']}, retrying...")
                    retry_times += 1
                else:
                    break

            if result is None:
                continue

            # 更新MongoDB中的文档状态
            collection.update_one(
                {'taskId': task_id},
                {'$set': {
                    'startTime': result['start_time'],
                    'finishTime': result['finish_time'],
                    'status': result['status']
                }},
            )
            print(f"task {task_id}: {result['info']}")

            webhook_url = f"{setting['webhook']}{updated_doc.get('webhook')}"
            headers = {'Content-Type': 'application/json'}
            if result['status'] == 3:
                if task_type == 'lora_train':
                    handle_lora_result(result, user_id, task_id, collection_result, webhook_url, setting)
                elif task_type == 'highres_task':
                    handle_highres_result(result, user_id, task_id, task_params, collection_result, webhook_url,
                                          setting)
            elif result['status'] == 4:
                data = {
                    'userId': user_id,
                    'taskId': task_id,
                    'finishTime': result['finish_time'].strftime("%Y-%m-%d %H:%M:%S"),
                    "taskTpye": "test",
                    "name": "test",
                    'status': False,  # Fail
                    'reason': result['info']
                }
                data = json.dumps(data, indent=4, default=str)
                response = requests.request("POST", webhook_url, data=data, headers=headers)
                print(f"webhook: {response}")
                # TODO warning failure
        else:
            print(f'Task {task_id} not found, pass')


def start_consumer(args):
    try:
        if isinstance(args, dict):
            arg1 = args['setting_file']
            arg2 = args['obj_store_setting']
        else:
            arg1 = args.setting_file
            arg2 = args.obj_store_setting
        # Load setting
        with open(arg1) as f:
            setting = json.load(f)
        with open(arg2) as f:
            obj_store_setting = json.load(f)
            setting['obj_store'] = obj_store_setting
    except Exception as e:
        print(f'Cannot load settings for scheduler: {e}')
        return

    # MongoDB配置
    MONGODB_SERVER = setting['mongo_server']
    DATABASE_NAME = setting['mongo_db_name']
    COLLECTION_NAME = setting['mongo_collection_name']

    # Kafka配置
    KAFKA_BOOTSTRAP_SERVERS = setting['kafka_server']
    KAFKA_TOPIC = setting['kafka_topic']

    # 创建消费者
    consumer = Consumer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS, 'group.id': 'mygroup'})
    consumer.subscribe([KAFKA_TOPIC])
    print(f"Listen to topic: {KAFKA_TOPIC}")

    # 创建MongoDB客户端
    client = MongoClient(MONGODB_SERVER)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    collection_result = db[f'{COLLECTION_NAME}_result']

    # 使用schedule库定时执行任务
    schedule.every(5).seconds.do(scan_and_do_task,
                                 consumer, collection, collection_result, setting)

    while True:
        schedule.run_pending()
        time.sleep(0.5)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Read from MongoDB and write to Kafka.")
    parser.add_argument("setting_file", type=str)
    parser.add_argument("obj_store_setting", type=str)
    args = parser.parse_args()

    start_consumer(args)
