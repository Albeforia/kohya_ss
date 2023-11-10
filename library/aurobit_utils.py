import os

import requests
from minio import Minio
from minio.error import S3Error


def create_minio_client(setting):
    provider = setting['provider']
    secret_id = setting['secret_id']
    secret_key = setting['secret_key']
    region = setting['region']
    if provider == 'tencent':
        # endpoint = format_endpoint(None, region, u'cos.', True, True)
        raise Exception()
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


def create_minio_client1():
    secret_id = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION')
    endpoint = f"s3.{region}.amazonaws.com"
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


def check_necessary_files():
    local_remote_mapping = [
        {
            'local': 'DFL/facelib/2DFAN.npy',
            'remote': 'weights/DFL/facelib/2DFAN.npy'
        },
        {
            'local': 'DFL/facelib/3DFAN.npy',
            'remote': 'weights/DFL/facelib/3DFAN.npy'
        },
        {
            'local': 'DFL/facelib/S3FD.npy',
            'remote': 'weights/DFL/facelib/S3FD.npy'
        },
        {
            'local': 'models/beautifulRealistic_brav5.safetensors',
            'remote': 'weights/models/beautifulRealistic_brav5.safetensors'
        },
    ]

    client = create_minio_client1()

    # Download scheduler_settings
    objects = client.list_objects(os.getenv('AWS_BUCKET'), prefix='yanjun/scheduler_settings/', recursive=False)
    if not os.path.exists('scheduler_settings'):
        os.makedirs('scheduler_settings', exist_ok=True)
        for obj in objects:
            _, filename = os.path.split(obj.object_name)
            if filename:
                client.fget_object(os.getenv('AWS_BUCKET'), obj.object_name,
                                   os.path.join('scheduler_settings', filename))

    for mapping in local_remote_mapping:
        local_path = mapping['local']
        remote_path = mapping['remote']

        if not os.path.exists(local_path):
            print(f"'{local_path}' does not exist. Downloading...")

            try:
                client.fget_object(
                    bucket_name=os.getenv('AWS_BUCKET'),
                    object_name=remote_path,
                    file_path=local_path)
                print(f"Downloaded to '{local_path}'")
            except Exception as e:
                print(f"Error occurred while downloading '{remote_path}': {str(e)}")
