
from __future__ import division
import warnings

from torch.utils.data.dataloader import DataLoader

# from core.utils.torch_util import worker_init_fn
from shaper.data import datasets as D
from shaper.data import transforms as T
from functools import partial
from dataset_prep.dataset import PartNetInsSegGlobal, collate
from core.utils.torch_util import worker_init_fn
from torch.utils.data.dataloader import default_collate


def build_dataloader(cfg, mode='train', zero_shot=False, cache_mode=False, shuffle_override = None, seen_only = False,
                     pose=False):
    assert mode in ['train', 'val', 'test']
    batch_size = cfg.TRAIN.BATCH_SIZE if mode == 'train' else cfg.TEST.BATCH_SIZE
    is_train = (mode == 'train')

    dataset = build_part_seg_dataset(cfg, split=mode, zero_shot=zero_shot, cache_mode=cache_mode, seen_only = seen_only, pose=pose)

    kwargs_dict = cfg.DATALOADER.KWARGS

    if cfg.DATALOADER.collate == 'partial':
        collate_fn = partial(collate,
                             num_centroids=kwargs_dict.num_centroids,
                             radius=kwargs_dict.radius,
                            num_neighbours=kwargs_dict.num_neighbours,
                            with_renorm=kwargs_dict.with_renorm,
                            with_resample=kwargs_dict.with_resample if is_train else False,
                            with_shift=kwargs_dict.with_shift if is_train else False,
                           sample_method=kwargs_dict.get('sample_method', 'FPS'))
    elif cfg.DATALOADER.collate == 'default':
        collate_fn = default_collate

    shuffle = shuffle_override if shuffle_override else is_train

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=(is_train and cfg.DATALOADER.DROP_LAST),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
    )

    return dataloader


def parse_augmentations(cfg, is_train=True):
    transform_list = []
    augmentations = cfg.TRAIN.AUGMENTATION if is_train else cfg.TEST.AUGMENTATION
    for aug in augmentations:
        if isinstance(aug, (list, tuple)):
            method = aug[0]
            args = aug[1:]
        else:
            method = aug
            args = []
        if cfg.INPUT.USE_NORMAL and hasattr(T, method + 'WithNormal'):
            method = method + 'WithNormal'
        transform_list.append(getattr(T, method)(*args))
    return transform_list


def build_part_seg_dataset(cfg, split='train', zero_shot=False, cache_mode=False, normalize=False, seen_only = False, pose=False):
    #split = cfg.DATASET[split.upper()]
    is_train = (split == 'train')

    transform_list = parse_augmentations(cfg, is_train)
    transform_list.insert(0, T.ToTensor())
    #transform_list.append(T.Transpose())
    transform = T.ComposeSeg(transform_list)

    dataset = PartNetInsSegGlobal(cfg.DATASET.path,
                                  split,
                                  train_objects = cfg.DATASET.train_objects,
                                  eval_objects = cfg.DATASET.eval_objects,
                                  zero_shot=zero_shot,
                                  cache_mode=cache_mode,
                                  normalize=normalize,
                                  transform=transform,
                                  downsample = cfg.DATASET.downsample,
                                  seen_only = seen_only,
                                  pose = pose)

    return dataset
