import argparse
import json
import sys
import time

import schedule
from confluent_kafka import Producer, KafkaException
from pymongo import MongoClient, ReturnDocument


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Read from MongoDB and write to Kafka.")
    parser.add_argument("setting_file", type=str)
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

    # 创建生产者
    producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})

    # 创建MongoDB客户端
    client = MongoClient(MONGODB_SERVER)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

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

    # 使用schedule库定时执行任务
    schedule.every(10).seconds.do(read_from_mongo_and_write_to_kafka)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
