import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np

from PIL import Image
from RAFT.core.utils import flow_viz

DEVICE = 'cuda'


def main(args):
    images = glob.glob(os.path.join(args.img_path, '*.png')) + \
             glob.glob(os.path.join(args.img_path, '*.jpg'))

    data_a = glob.glob(os.path.join(args.path_a, '*.npy'))
    data_b = glob.glob(os.path.join(args.path_b, '*.npy'))

    images = sorted(images)
    data_a = sorted(data_a)
    data_b = sorted(data_b)
    assert len(data_a) == len(data_b)

    for i, data_file in enumerate(data_a):
        img = np.array(Image.open(images[i])).astype(np.uint8)

        flow_a = np.transpose(np.squeeze(np.load(data_file)), [2, 1, 0])
        flow_b = np.transpose(np.squeeze(np.load(data_b[i])), [2, 1, 0])

        diff = np.abs(flow_a - flow_b)
        diff_img = flow_viz.flow_to_image(diff)
        # cv2.imshow('image', diff_img[:, :, [2, 1, 0]] / 255.0)

        flow_ab = np.concatenate([flow_a, flow_b, diff], axis=0)
        cv2.imshow('image', flow_viz.flow_to_image(flow_ab)[:, :, [2, 1, 0]] / 255.0)
        cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
    parser.add_argument('--path_a')
    parser.add_argument('--path_b')
    args = parser.parse_args()

    main(args)
