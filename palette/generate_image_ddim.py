from pathlib import Path
import argparse

from tqdm import trange
import numpy as np
import cv2

import torch
from torchvision import transforms

from palette.samplers import DDIMSampler
import palette.core.parser as Parser
from palette.utils import read_images
from core.logger import VisualWriter, InfoLogger
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric


def run_ddim_prediction(args, sampler, trans, img_hubble: np.ndarray):
    """Run DDIM prediction
    Saves the generated images to the output path

    Args:
        args: command line arguments
        sampler: sampler network
        trans: image transformation
        img_hubble: input Hubble image
    """

    size = args.crop_size
    stride = args.stride
    batch_size = args.batch
    n_iter = args.n_iter  # number of iterations per image

    img_webb_gen = np.zeros_like(img_hubble)
    img_map = np.zeros_like(img_hubble)

    h, w = img_hubble.shape[:2]

    batch = []
    pos = []

    for i in trange(0, h - size + stride, stride):
        for j in range(0, w - size + stride, stride):

            for _ in trange(n_iter):
                img_hubble_crop = img_hubble[i:i + size, j:j + size, :].copy()
                img_map[i:i + size, j:j + size, :] += 1
                img_hubble_crop_torch = trans(img_hubble_crop).cuda()
                img_hubble_crop_torch = img_hubble_crop_torch.unsqueeze(0)
                batch.append(img_hubble_crop_torch)
                pos.append([i, j])

                if len(batch) == batch_size or (i == h - size and j == w - size):
                    batch = torch.cat(batch, 0)

                    with torch.no_grad():
                        img_webb_gen_crop, interm = sampler.sample(S=args.n_sample, conditioning=batch,
                                                                   batch_size=batch.shape[0],
                                                                   shape=[3, size, size],
                                                                   verbose=False, log_every_t=1)
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

    out_parent = Path(args.output_path).parent
    out_parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output_path, img_webb_gen, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to the config file.')
    parser.add_argument('--model_path', '-m', type=str, required=True,
                        help='Path to the model.')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the directory containing the input images.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output Webb image.')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--n_sample', type=int, default=100)

    args = parser.parse_args()

    args.phase = 'test'
    opt = Parser.parse(args)

    # set model path
    opt['path']['resume_state'] = args.model_path

    # set logger
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)

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

    net_g = model.netG.cuda()
    net_g.set_new_noise_schedule(phase='test')
    net_g_ema = model.netG_EMA.cuda()
    net_g_ema.set_new_noise_schedule(phase='test')

    if args.use_ema:
        sampler = DDIMSampler(net_g_ema)
    else:
        sampler = DDIMSampler(net_g)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    stride = args.stride
    size = args.crop_size

    img_hubble, img_webb = read_images(args.input_path, size, stride)
    run_ddim_prediction(args, sampler, trans, img_hubble)
