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
from collections import Counter
from PIL import Image
from pathlib import Path
from deepface import DeepFace

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()


def get_image_file_paths(directory):
    image_file_paths = []
    for file in os.listdir(directory):
        full_file_path = os.path.join(directory, file)
        if os.path.isfile(full_file_path):
            # Check the file extension
            _, ext = os.path.splitext(full_file_path)
            if ext.lower() in ['.jpg', '.jpeg', '.png']:
                image_file_paths.append(full_file_path)
    return image_file_paths


def parse_analysis_result(result):
    if not result:
        log.warning("No face detected in the image.")
        return None

    first_face = result[0]
    parsed_result = {
        "age": int(first_face["age"]),
        "gender": first_face["dominant_gender"],
        "race": first_face["dominant_race"]
    }

    return parsed_result


def analyze_face_data(face_data):
    if not face_data:
        return {}

    # Compute average age
    average_age = sum(face['age'] for face in face_data) / len(face_data)

    # Compute gender ratio for the most common gender
    gender_counter = Counter(face['gender'] for face in face_data)
    total_gender_count = sum(gender_counter.values())
    most_common_gender, most_common_gender_count = gender_counter.most_common(1)[0]
    gender_ratio = most_common_gender_count / total_gender_count

    # Compute race ratio for the most common race
    race_counter = Counter(face['race'] for face in face_data)
    total_race_count = sum(race_counter.values())
    most_common_race, most_common_race_count = race_counter.most_common(1)[0]
    race_ratio = most_common_race_count / total_race_count

    return {
        "total_faces": len(face_data),
        "average_age": average_age,
        "most_common_gender": most_common_gender,
        # "gender_ratio": gender_ratio,
        "most_common_race": most_common_race,
        # "race_ratio": race_ratio
    }


def gather_face_info(input_path):
    files = get_image_file_paths(input_path)
    result = []
    for img in files:
        face_data = parse_analysis_result(DeepFace.analyze(img, ('race', 'gender', 'age'), enforce_detection=False))
        if not face_data:
            continue
        face_data['source'] = img
        result.append(face_data)

    return (result, analyze_face_data(result))
