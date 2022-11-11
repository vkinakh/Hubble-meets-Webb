![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)

# Hubble meets Webb:  image-to-image translation in astronomy 

This repo contains official Pytorch implementation of the paper: **Hubble meets Webb:  image-to-image translation in astronomy**

[Interactive demo](https://hubble-to-webb.herokuapp.com/)

# Contents
1. [Abstract](#abstract)
2. [Installation](#installation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Results](#results)
6. [Citation](#citation)

## Abstract

We present a study on the image-to-image translation problem for the prediction of future satellite Webb images from the available Hubble images. In the paper, we compare multiple image-to-image translation setups Pix2Pix (paired setup), CycleGAN (unpaired setup), and denoising diffusion probabilistic model (DDPM)-based Palette (paired setup). We introduce a novel image-to-image translation framework Turbo that generalizes several state-of-the-art methods in both paired families such as Pix2Pix and unpaired such as CycleGAN. We demonstrate the importance of synchronization between image pairs in image-to-image translation problems. We introduce a new framework for practically measuring uncertainty in image-to-image translation problems by utilizing the inherent stochasticity of the DDPM-based Palette method. We compare the methods using multiple metrics such as MSE, SSIM, PSNR, LPIPS, and FID. 

## Installation

### Conda installation
```commandline
conda env create -f environment.yml
```

## Training

### Training of the Pix2Pix/CycleGAN/Turbo models

In order to train Pix2Pix/CycleGAN/Turbo based models use `turbo/train.py` model. Specify model name with `--model` parameter.

Models and their codes for training

| Model                           |Code|
|---------------------------------|----|
| L1 loss | pix2pixl1 |
| L2 loss | pix2pixl2 |
| Pix2Pix (L1 loss + GAN)         | pix2pix|
| CycleGAN (Cycle losses + GANs)  | cycle_gan|  
| Elastic net (L1 + L2 losses)    | pix2pix_elastic_net|
| Pix2Pix (L1 loss + GAN) + LPIPS |pix2pix_l1_gan_lpips| 
| L2 loss + LPIPS |pix2pix_lpips |
| Turbo | turbo|
| Turbo one cycle| turbo_one_cycle|

Make sure to always specify `--data=space`. Look into `turbo/options` and `turbo/models` for other commandline training/testing options.

### Training of the Palette (DDPM based image to image translation model)

Fill the config file, as it is shown in `configs/palette/hubble2webb_ddpm.json`.

Then run:
```commandline
python palette/train_palette.py --config=<path to your config file> --phase=train --batch=<batch size> 
```
Check for other parameters in `palette/train_palette.py`

## Evaluation

### Generate images using Pix2Pix/CycleGAN/Turbo models
To generate Webb image from Hubble image using Pix2Pix/CycleGAN/Turbo models use `turbo/generate_image.py`. See the commandline options `turbo/options`

### Generate images using Palette DDPM model (with default number of steps)
To generate Webb image from Hubble image using Palette DDPM model first fill the config as shown in `configs/palette/hubble2webb_ddim.json` and then use `palette/generate_image_ddpm.py`. 
`--input_path` should be path to directory with 1 Webb image named `webb*` and 1 Hubble image named `hubble*`.
`--output_path` should be `png` image.
`--model_path` should path to the directory + file stem (name without extension), which should contain 3 files: `<stem>.state`, `<stem>_Network.pth`, `<stem>_Network_ema.pth`

```commandline
python palette/generate_image_ddpm.py \ 
--config=<path to your config> \
--input_path=<path to dir with images>  \
--output_path=<output path of gen image> \
--model_path=<path to model> \
--crop_size=<size of the image> \
--stride=<stride to use when images are generated> \
--batch=<batch size> \
--use_ema - if added, then EMA version of the model is used
```

### Generate images using DDIM sampler (number of denoising steps can be changed)

To generate Webb image from Hubble image using Palette with DDIM sampler first fill the config as shown in `configs/palette/hubble2webb_ddim.json` and the use `palette/generate_image_ddim.py`.
`--input_path` should be path to directory with 1 Webb image named `webb*` and 1 Hubble image named `hubble*`.
`--output_path` should be `png` image.
`--model_path` should path to the directory + file stem (name without extension), which should contain 3 files: `<stem>.state`, `<stem>_Network.pth`, `<stem>_Network_ema.pth`
```commandline
python palette/generate_image_ddim.py \ 
--config=<path to your config> \
--input_path=<path to dir with images>  \
--output_path=<output path of gen image> \
--model_path=<path to model> \
--crop_size=<size of the image> \
--stride=<stride to use when images are generated> \
--batch=<batch size> \
--n_sample=<number of denoising step, more better quality> \
--use_ema - if added, then EMA version of the model is used
```

### To compute metrics for generated image

To compute MSE, SSIM, PSNR, LPIPS and FID metrics for generated images run:
```commandline
python compute_metrics.py \
--img_target=<path to target Webb image> \
--img_output=<path to predicted Webb image> \
--path_fid=<path to dir where images for FID calculation are saved>
```

## Results

### CycleGAN unpaired setup
|Method|MSE|SSIM|PSNR|LPIPS|FID|
|------|---|----|----|-----|---|
| CycleGan| 0.009747 | 0.8257 | 20.11 | 0.48 | 128.12| 

### Manual synchronization
|Method|MSE|SSIM|PSNR|LPIPS|FID|
|-----|---|----|----|-----|---|
|Pix2Pix|0.007287|0.865 | 21.37 | 0.5 |102.61|
|Turbo|0.008184|0.851|20.87|0.49|98.41|
|DDPM (Palette)|0.002910|0.8779|25.36|0.43|51.2|

### Global synchronization
|Method|MSE|SSIM|PSNR|LPIPS|FID|
|-----|---|----|----|-----|---|
|Pix2Pix|0.002601|0.9174|25.85|0.46|55.69|
|Turbo|0.003104|0.9084|25.08|0.45|48.57|
|DDPM (Palette)|0.00154|0.9367|28.12|0.45|43.97|

### Local synchronization
| Method                           | MSE          | SSIM      | PSNR      | LPIPS    | FID       |
|----------------------------------|--------------|-----------|-----------|----------|-----------|
| L1                               | 0.002023     | 0.9348    | 26.94     | 0.47     | 83.32     |
| L2                               | 0.002003     | 0.9352    | 26.98     | 0.47     | 76.03     |
| Elastic net(L1+L2)               | 0.002028     | 0.9343    | 26.93     | 0.47     | 82.71     |
| L1+LPIPS                         | 0.002147     | 0.9302    | 26.68     | 0.44     | 72.84     |
| Pix2Pix                          | 0.002099     | 0.9322    | 26.78     | 0.44     | 54.58     |
| Pix2Pix+LPIPS                    | 0.001985     | 0.9345    | 27.02     | 0.44     | 58.86     |
| Turbo                            | 0.002582     | 0.9171    | 25.88     | 0.41     | 43.36     |
| Turbo+LPIPS                      | 0.002564     | 0.9181    | 25.91     | 0.39     | 50.83     |
| Turbo one cycle(L^reverse)       | 0.002425     | 0.9281    | 26.15     | 0.45     | 70.51     |
| Turbo one cycle(L^reverse)+LPIPS | 0.002436     | 0.9281    | 26.13     | 0.46     | 67.52     |
| Turbo same D                     | 0.002488     | 0.9165    | 26.04     | 0.4      | 55.29     |
| Turbo same D+LPIPS               | 0.002439     | 0.9187    | 26.13     | **0.39** | 55.88     |
| **DDPM (Palette)**               | **0.001224** | **0.948** | **29.12** | 0.44     | **30.08** |

### Weights
Coming soon

## Citation
Coming soon