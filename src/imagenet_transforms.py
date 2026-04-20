from __future__ import annotations

import torchvision.transforms as T


def imagenet_normalize() -> T.Normalize:
    return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def build_train_transforms(input_size: int, use_random_resized_crop: bool) -> T.Compose:
    norm = imagenet_normalize()
    if use_random_resized_crop:
        aug = T.Compose(
            [
                T.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                norm,
            ]
        )
    else:
        aug = T.Compose(
            [
                T.Resize(int(round(input_size / 224.0 * 256)) or 256),
                T.RandomCrop(input_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                norm,
            ]
        )
    return aug


def build_eval_transforms(input_size: int) -> T.Compose:
    resize = int(round(input_size / 224.0 * 256)) or 256
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(input_size),
            T.ToTensor(),
            imagenet_normalize(),
        ]
    )
