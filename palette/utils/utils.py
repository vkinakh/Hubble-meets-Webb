from typing import Tuple
from pathlib import Path

import numpy as np
import cv2


def read_images(dataroot: str, size: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reads Hubble and Webb image from a given dataroot
    resizes to bigger size
    adds padding of the size half of crop size
    adds padding to make it divisible by stride

    Args:
        dataroot: path to the folder containing Hubble and Webb images
        size: crop size
        stride: stride

    Returns:
        Tuple of Hubble and Webb images
    """

    dataroot = Path(dataroot)

    path_webb = [p for p in dataroot.glob('webb*')][0]
    path_hubble = [p for p in dataroot.glob('hubble*')][0]

    img_webb = cv2.imread(str(path_webb))
    h_webb, w_webb, _ = img_webb.shape

    img_hubble = cv2.imread(str(path_hubble))
    h_hubble, w_hubble, _ = img_hubble.shape

    h = max(h_webb, h_hubble)
    w = max(w_webb, w_hubble)
    img_webb = cv2.resize(img_webb, (w, h))
    img_hubble = cv2.resize(img_hubble, (w, h))

    # add padding
    pad_size = size // 2
    img_webb = cv2.copyMakeBorder(img_webb, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
    img_hubble = cv2.copyMakeBorder(img_hubble, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)

    h, w = img_webb.shape[:2]
    if h % stride != 0:
        h_pad = (h // stride + 1) * stride
        w_pad = (w // stride + 1) * stride
    else:
        h_pad = h
        w_pad = w

    im_webb_pad = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
    im_webb_pad[:h, :w, :] = img_webb
    img_webb = im_webb_pad

    im_hubble_pad = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
    im_hubble_pad[:h, :w, :] = img_hubble
    img_hubble = im_hubble_pad

    img_webb = cv2.cvtColor(img_webb, cv2.COLOR_BGR2RGB)
    img_hubble = cv2.cvtColor(img_hubble, cv2.COLOR_BGR2RGB)

    img_webb = img_webb / 255.
    img_webb = img_webb.astype(np.float32)

    img_hubble = img_hubble / 255.
    img_hubble = img_hubble.astype(np.float32)
    return img_hubble, img_webb
