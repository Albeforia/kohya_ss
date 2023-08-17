import argparse
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
from deepface import DeepFace
from pathlib import Path


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


def _parse_analysis_result(result):
    if not result:
        print("No face detected in the image.")
        return None

    first_face = result[0]
    man_prob = first_face['gender']['Man']
    woman_prob = first_face['gender']['Woman']
    dom_gender = first_face['dominant_gender']
    ratio = woman_prob / man_prob if dom_gender == 'Woman' else man_prob / woman_prob
    parsed_result = {
        # "age": int(first_face["age"]),
        "gender": dom_gender,
        "probability_radio": ratio,
        "race": first_face["dominant_race"]
    }

    return parsed_result


def _analyze_face_data(face_data):
    if not face_data:
        return {}

    # Compute average age
    # average_age = sum(face['age'] for face in face_data) / len(face_data)

    # Compute gender ratio for the most common gender
    gender_counter = Counter(face['gender'] for face in face_data)
    total_gender_count = sum(gender_counter.values())
    most_common_gender, most_common_gender_count = gender_counter.most_common(1)[0]
    gender_ratio = most_common_gender_count / total_gender_count

    gender = ""
    if gender_ratio > 0.8:
        gender = most_common_gender
    else:
        probability_radios = {}
        for gender_type, count in gender_counter.items():
            probability_radios[gender_type] = sum(
                d['probability_radio'] for d in face_data if d['gender'] == gender_type) / count
        gender = max(probability_radios, key=probability_radios.get)
        print(f"By frequency: {most_common_gender}; by weight: {gender}")

    # Compute race ratio for the most common race
    race_counter = Counter(face['race'] for face in face_data)
    total_race_count = sum(race_counter.values())
    most_common_race, most_common_race_count = race_counter.most_common(1)[0]
    race_ratio = most_common_race_count / total_race_count

    return {
        "total_faces": len(face_data),
        # "average_age": average_age,
        "most_common_gender": gender,
        # "gender_ratio": gender_ratio,
        "most_common_race": most_common_race,
        # "race_ratio": race_ratio
    }


def gather_face_info(input_path):
    files = _get_image_file_paths(input_path)
    result = []
    for img in files:
        face_data = _parse_analysis_result(DeepFace.analyze(img, ('race', 'gender'), enforce_detection=False))
        if not face_data:
            continue
        face_data['source'] = img
        result.append(face_data)

    return (result, _analyze_face_data(result))


def main(args):
    results, stats = gather_face_info(args.input_path)
    with open(f'{args.output_path}/faces.txt', 'w') as f:
        f.write(json.dumps({
            'faces': results,
            'stats': stats
        }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_path',
        dest='input_path',
        type=str,
        default=None)
    parser.add_argument(
        '--output_path',
        dest='output_path',
        type=str,
        default=None)

    main(parser.parse_args())
