
import math
import random
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder, folder


class MaskGenerator(object):
    def __init__(self, input_size=256, mask_patch_size=16, model_patch_size=2, mask_ratio=0.5):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class TrainTransform(object):
    def __init__(self, configs):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop((256, 256), scale=(0.8, 1.2), ratio=(4. / 5., 5. / 4.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

        self.mask_generator = MaskGenerator(
            input_size=256,
            mask_patch_size=16,
            model_patch_size=configs.patch_size,
            mask_ratio=0.5,
        )

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def get_loader(configs):
    train_transform = TrainTransform(configs)
    train_dataset = ImageFolder(configs.datadir, train_transform)

    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)

    num_workers = 16
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=configs.batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader


