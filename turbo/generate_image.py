from typing import Tuple
from pathlib import Path

import cv2
import numpy as np
from tqdm import trange

import torch
from torchvision import transforms

from models import create_model
from options.test_options import TestOptions


def read_images(dataroot: str, stride: int) -> Tuple[np.ndarray, np.ndarray]:
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
    h_pad = (h // stride + 1) * stride
    w_pad = (w // stride + 1) * stride

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


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    size = 256
    stride = opt.stride

    if opt.model in ['pix2pix', 'pix2pix_l1_gan_lpips']:
        net_g = model.netG
    else:
        # cycle gan
        net_g = model.netG_A

    net_g.eval()

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    img_hubble, img_webb = read_images(opt.dataroot, stride)
    img_webb_gen = np.zeros_like(img_webb)
    img_map = np.zeros_like(img_webb)

    h, w = img_webb.shape[:2]

    for i in trange(0, h - size + stride, stride):
        for j in range(0, w - size + stride, stride):
            img_hubble_crop = img_hubble[i:i+size, j:j+size, :].copy()
            img_map[i:i+size, j:j+size, :] += 1

            img_hubble_crop_torch = trans(img_hubble_crop)
            img_hubble_crop_torch = img_hubble_crop_torch.unsqueeze(0)

            with torch.no_grad():
                img_webb_gen_crop = net_g(img_hubble_crop_torch)

            img_webb_gen_crop_np = img_webb_gen_crop.squeeze().detach().cpu().numpy()
            img_webb_gen_crop_np = np.transpose(img_webb_gen_crop_np, (1, 2, 0))
            img_webb_gen_crop_np = img_webb_gen_crop_np * 0.5 + 0.5

            img_webb_gen[i:i+size, j:j+size, :] += img_webb_gen_crop_np

    img_webb_gen = img_webb_gen / img_map
    img_webb_gen = img_webb_gen * 255.
    img_webb_gen = img_webb_gen.astype(np.uint8)
    img_webb_gen = cv2.cvtColor(img_webb_gen, cv2.COLOR_RGB2BGR)

    cv2.imwrite(opt.savename, img_webb_gen, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    if opt.display:
        cv2.namedWindow('gen_webb', cv2.WINDOW_NORMAL)
        cv2.imshow('gen_webb', img_webb_gen)

        cv2.namedWindow('hubble', cv2.WINDOW_NORMAL)
        cv2.imshow('hubble', img_hubble)

        cv2.namedWindow('webb', cv2.WINDOW_NORMAL)
        cv2.imshow('webb', img_webb)
        cv2.waitKey(0)
