import os
from torchvision import datasets, transforms
from dataloader.Eyepacs import Eyepacs
from dataloader.Messidor import Messidor1Dataset,Messidor2Dataset
from dataloader.Apots import Apots
from dataloader.RFMid import RFMiD
from dataloader.Rsnr import RSNR
from dataloader.MICCAI import MICCAI
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
    resize_im = args.input_size > 32
    input_size = args.input_size
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=min(args.color_jitter, 0.5),  # Garantir que hue_factor não ultrapasse 0.5
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
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