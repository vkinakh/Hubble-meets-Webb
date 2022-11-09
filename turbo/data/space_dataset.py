from pathlib import Path

import numpy as np
import cv2

from torchvision import transforms

from .base_dataset import BaseDataset


class SpaceDataset(BaseDataset):

    def __init__(self, opt):

        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        dataroot = Path(opt.dataroot)

        self.paths_webb = sorted([str(x) for x in dataroot.glob('webb*')])
        self.paths_hubble = sorted([str(x) for x in dataroot.glob('hubble*')])

        self.imgs_webb = [cv2.imread(x) for x in self.paths_webb]
        self.imgs_webb = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in self.imgs_webb]
        self.imgs_webb = [(x / 255).astype(np.float32) for x in self.imgs_webb]

        self.imgs_hubble = [cv2.imread(x) for x in self.paths_hubble]
        self.imgs_hubble = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in self.imgs_hubble]
        self.imgs_hubble = [(x / 255).astype(np.float32) for x in self.imgs_hubble]

        self.size = 256

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return 20_000

    def __getitem__(self, index):

        n = index % len(self.paths_webb)
        img_webb = self.imgs_webb[n]
        img_hubble = self.imgs_hubble[n]

        h, w, _ = img_webb.shape
        y = np.random.randint(0, h - self.size)
        x = np.random.randint(0, w - self.size)

        img_webb_crop = img_webb[y:y+self.size, x:x+self.size, :]
        img_hubble_crop = img_hubble[y:y+self.size, x:x+self.size, :]

        if np.random.rand() > 0.5:
            img_webb_crop = np.fliplr(img_webb_crop)
            img_hubble_crop = np.fliplr(img_hubble_crop)
        if np.random.rand() > 0.5:
            img_webb_crop = np.flipud(img_webb_crop)
            img_hubble_crop = np.flipud(img_hubble_crop)
        if np.random.rand() > 0.5:
            img_webb_crop = np.rot90(img_webb_crop)
            img_hubble_crop = np.rot90(img_hubble_crop)

        img_webb_crop = self.transform(img_webb_crop.copy())
        img_hubble_crop = self.transform(img_hubble_crop.copy())

        return {
            'A': img_webb_crop,
            'B': img_hubble_crop,
            'A_paths': str(self.paths_webb[n]),
            'B_paths': str(self.paths_hubble[n])
        }
