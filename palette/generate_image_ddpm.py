from pathlib import Path
import argparse

from tqdm import trange
import numpy as np
import cv2

import torch
from torchvision import transforms

import palette.core.parser as Parser
from palette.utils import read_images
from core.logger import VisualWriter, InfoLogger
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric


def run_ddpm_prediction(args, net_g, trans, img_hubble: np.ndarray):
    """Run DDPM prediction
    Saves the generated images to the output path

    Args:
        args: command line arguments
        net_g: generator network
        trans: image transformation
        img_hubble: input Hubble image
    """

    size = args.crop_size
    stride = args.stride
    batch_size = args.batch

    out_path_dir = Path(args.output_path).parent
    out_path_dir.mkdir(parents=True, exist_ok=True)

    img_webb_gen = np.zeros_like(img_hubble)
    img_map = np.zeros_like(img_hubble)

    h, w = img_hubble.shape[:2]

    batch = []
    pos = []

    for i in trange(0, h - size + stride, stride):
        for j in trange(0, w - size + stride, stride):
                img_hubble_crop = img_hubble[i:i + size, j:j + size, :].copy()
                img_map[i:i + size, j:j + size, :] += 1
                img_hubble_crop_torch = trans(img_hubble_crop).cuda()
                img_hubble_crop_torch = img_hubble_crop_torch.unsqueeze(0)
                batch.append(img_hubble_crop_torch)
                pos.append([i, j])

                if len(batch) == batch_size or (i == h - size and j == w - size):
                    batch = torch.cat(batch, 0)

                    with torch.no_grad():
                        img_webb_gen_crop, visual = net_g.restoration(batch, display=True)

                    for k in range(len(batch)):
                        curr_i, curr_j = pos[k]

                        img_webb_gen_crop_np = img_webb_gen_crop[k].squeeze().detach().cpu().numpy()
                        img_webb_gen_crop_np = np.transpose(img_webb_gen_crop_np, (1, 2, 0))
                        img_webb_gen_crop_np = img_webb_gen_crop_np * 0.5 + 0.5
                        img_webb_gen_crop_np = img_webb_gen_crop_np.clip(0, 1)
                        img_webb_gen[curr_i:curr_i + size, curr_j:curr_j + size, :] += img_webb_gen_crop_np

                    batch = []
                    pos = []

    img_webb_gen = img_webb_gen / img_map
    img_webb_gen = img_webb_gen * 255.
    img_webb_gen = img_webb_gen.astype(np.uint8)
    img_webb_gen = cv2.cvtColor(img_webb_gen, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_path, img_webb_gen, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config file.')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to directory containing the input images')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output image')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_ema', action='store_true')

    args = parser.parse_args()
    args.phase = 'test'
    opt = Parser.parse(args)

    # set model path
    opt['path']['resume_state'] = args.model_path

    # set logger
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    # set networks and dataset
    phase_loader, val_loader = define_dataloader(phase_logger, opt)  # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    # set metrics, loss, optimizer and  schedulers
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt=opt,
        networks=networks,
        phase_loader=phase_loader,
        val_loader=val_loader,
        losses=losses,
        metrics=metrics,
        logger=phase_logger,
        writer=phase_writer
    )

    if args.use_ema:
        net_g = model.netG_EMA
    else:
        net_g = model.netG

    net_g.set_new_noise_schedule(phase='test')
    net_g.cuda()

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    stride = args.stride
    crop_size = args.crop_size

    img_hubble, img_webb = read_images(args.input_path, crop_size, stride)
    run_ddpm_prediction(args, net_g, trans, img_hubble)
