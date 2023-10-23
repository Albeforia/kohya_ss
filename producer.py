import argparse
import json
import sys
import time

import pynvml
import schedule
import sentry_sdk
from confluent_kafka import Producer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
from pymongo import MongoClient, ReturnDocument


def create_topic(server, topic):
    # 创建一个AdminClient
    a = AdminClient({'bootstrap.servers': server})

    # 定义一个新的topic
    new_topics = [NewTopic(topic, num_partitions=1, replication_factor=1)]

    # 创建topic
    fs = a.create_topics(new_topics)

    # 等待每个Future完成
    for topic, f in fs.items():
        try:
            f.result()  # The result itself is None
            print("Topic {} created".format(topic))
        except Exception as e:
            print("Failed to create topic {}: {}".format(topic, e))


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Read from MongoDB and write to Kafka.")
    parser.add_argument("setting_file", type=str)
    parser.add_argument('--interval', dest='interval', type=float, default=1)
    args = parser.parse_args()

    # Load setting
    with open(args.setting_file) as f:
        setting = json.load(f)

    # MongoDB配置
    MONGODB_SERVER = setting['mongo_server']
    DATABASE_NAME = setting['mongo_db_name']
    COLLECTION_NAME = setting['mongo_collection_name']

    # Kafka配置
    KAFKA_BOOTSTRAP_SERVERS = setting['kafka_server']
    KAFKA_TOPIC = setting['kafka_topic']

    # Sentry
    sentry_sdk.init(
        dsn=setting['sentry_dsn'],
    )

    # 创建生产者
    producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})

    create_topic(KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC)

    # 创建MongoDB客户端
    client = MongoClient(MONGODB_SERVER)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    #
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # 写入消息到Kafka
    def write_message(message, doc_id):
        try:
            producer.produce(KAFKA_TOPIC, message)
            producer.flush()
            print(f'Successfully written message {message} to {KAFKA_TOPIC}')
        except KafkaException as e:
            print(f'Failed to write message {message} to {KAFKA_TOPIC}: {e}')
            sys.stderr.write('%% %s\n' % e)
            # Kafka写入失败，将MongoDB中对应文档的状态设为0
            collection.update_one({'_id': doc_id}, {'$set': {'status': 0}})

    # 从MongoDB读取数据，并写入Kafka
    def read_from_mongo_and_write_to_kafka():
        utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        # print("GPU Compute Utilization: {}%".format(utilization.gpu))
        if utilization.gpu > 80:
            # print("Skip because of high GPU utilization")
            return

        try:
            doc = collection.find_one_and_update(
                {'status': 0, 'taskType': setting['task_type']},
                {'$set': {'status': 1}},
                return_document=ReturnDocument.AFTER
            )
            if doc is not None:
                task_id = doc.get('taskId')
                if task_id:
                    print('Found doc, try writing...')
                    write_message(task_id, doc['_id'])
        except Exception as e:
            print(f"{e} happened when accessing MongoDB")

    # 使用schedule库定时执行任务
    schedule.every(args.interval).minutes.do(read_from_mongo_and_write_to_kafka)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
