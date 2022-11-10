from typing import Tuple
from pathlib import Path
from argparse import ArgumentParser
import random

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from cleanfid import fid

import torch
import lpips


def compute_metrics(img1: np.ndarray, img2: np.ndarray) -> Tuple:
    mse_ = mse(img1, img2)
    ssim_ = ssim(img1, img2, multichannel=True)
    psnr_ = psnr(img1, img2)
    return mse_, ssim_, psnr_


def compute_lpips(img1: np.ndarray, img2: np.ndarray) -> float:
    loss_fn_vgg = lpips.LPIPS(net='vgg')

    img1 = (img1 - 0.5) / 0.5
    img1 = img1.astype(np.float32)
    img2 = (img2 - 0.5) / 0.5
    img2 = img2.astype(np.float32)

    lpips_vals = []
    size = 256
    stride = 128

    h, w = img1.shape[:2]

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            img1_crop = img1[i:i+size, j:j+size]
            img2_crop = img2[i:i+size, j:j+size]

            if img1_crop.shape[0] < size:
                continue

            lpips_vals.append(loss_fn_vgg(torch.from_numpy(img1_crop).permute(2, 0, 1).unsqueeze(0),
                                          torch.from_numpy(img2_crop).permute(2, 0, 1).unsqueeze(0)).item())
    return np.mean(lpips_vals)


def crop_samples(input_path: str, output_dir: Path,
                 crop_size: int = 299, n_samples: int = 10_000) -> None:

    img = cv2.imread(input_path)
    h, w = img.shape[:2]

    for i in range(n_samples):
        y = np.random.randint(0, h - crop_size)
        x = np.random.randint(0, w - crop_size)

        img_crop = img[y:y+crop_size, x:x+crop_size, :]

        if random.random() > 0.5:
            img_crop = cv2.flip(img_crop, 1)

        if random.random() > 0.5:
            img_crop = cv2.flip(img_crop, 0)

        if random.random() > 0.5:
            img_crop = cv2.flip(img_crop, -1)

        cv2.imwrite(str(output_dir / f'{i+1:06}.jpg'), img_crop)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img_target', type=str, required=True, help='Path to target image')
    parser.add_argument('--img_output', type=str, required=True, help='Path to generated image')
    parser.add_argument('--path_fid', type=str, required=True,
                        help='Path to the folder, where to save images before FID calculation')
    args = parser.parse_args()

    path_webb = args.img_target
    path_webb_pred = args.img_output

    im_webb = cv2.imread(path_webb)
    im_webb_pred = cv2.imread(path_webb_pred)

    # Crop samples for FID calculation
    fid_dir = Path(args.path_fid)
    fid_dir_target = fid_dir / 'target'
    fid_dir_target.mkdir(parents=True, exist_ok=True)
    fid_dir_output = fid_dir / 'output'
    fid_dir_output.mkdir(parents=True, exist_ok=True)

    crop_samples(path_webb, fid_dir_target)
    crop_samples(path_webb_pred, fid_dir_output)

    h, w = im_webb.shape[:2]
    im_webb_pred = im_webb_pred[:h, :w, :]

    im_webb = cv2.cvtColor(im_webb, cv2.COLOR_BGR2RGB)
    im_webb = im_webb.astype(np.float32) / 255.0
    im_webb_pred = cv2.cvtColor(im_webb_pred, cv2.COLOR_BGR2RGB)
    im_webb_pred = im_webb_pred.astype(np.float32) / 255.0

    mse_, ssim_, psnr_ = compute_metrics(im_webb, im_webb_pred)
    print(f'MSE: {mse_:.9f}')
    print(f'SSIM: {ssim_:.4f}')
    print(f'PSNR: {psnr_:.2f}')

    # compute lpips
    lpips_ = compute_lpips(im_webb, im_webb_pred)
    print(f'LPIPS: {lpips_:.4f}')

    # compute FID
    fid_ = fid.compute_fid(str(fid_dir_target), str(fid_dir_output))
    print(f'FID: {fid_:.2f}')

    # compute FID CLIP
    fid_clip_ = fid.compute_fid(str(fid_dir_target), str(fid_dir_output), model_name='clip_vit_b_32')
    print(f'FID CLIP: {fid_clip_:.2f}')
