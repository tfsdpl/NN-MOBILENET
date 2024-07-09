import os
from torchvision import datasets, transforms
from dataloader.Apots import Apots
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from torchvision.transforms import InterpolationMode
import config

def build_dataset(is_train):
    transform = build_transform(is_train)
    dataset = Apots(image_dir='dataset/APTOS/train', label_dir='dataset/APTOS/train.csv', transform=transform)
    return dataset

def build_transform(is_train):
    imagenet_default_mean_and_std = False
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform(
            input_size=244,
            is_training=True,
            color_jitter=min(0.4, 0.5),  # Garantir que hue_factor não ultrapasse 0.5
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
        return transform
    else:
        t = []
        size = int(config.IMPUT_SIZE/ config.CROP_PCT)
        t.append(
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        )
        t.append(transforms.CenterCrop(config.IMPUT_SIZE))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)