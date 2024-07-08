import os
from torchvision import datasets, transforms
from dataloader.Apots import Apots
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from torchvision.transforms import InterpolationMode

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    dataset = Apots(image_dir='dataset/APTOS/train', label_dir='dataset/APTOS/train.csv', transform=transform)
    nb_classes = args.nb_classes



    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = True
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform(
            input_size=244,
            is_training=True,
            color_jitter=min(args.color_jitter, 0.5),  # Garantir que hue_factor não ultrapasse 0.5
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
        if resize_im:
            if args.input_size >= 384:
                t.append(
                    transforms.Resize((args.input_size, args.input_size),
                                      interpolation=InterpolationMode.BICUBIC),
                )
                print(f"Warping {args.input_size} size input images...")
            else:
                if args.crop_pct is None:
                    args.crop_pct = 224 / 256
                size = int(args.input_size / args.crop_pct)
                t.append(
                    transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
                )
                t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)