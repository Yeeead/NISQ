from __future__ import annotations

from typing import Optional, Tuple

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from configs.default import DataConfig


def _maybe_subset(dataset: Dataset, limit: int) -> Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def _build_transform(
    image_size: int,
    normalize_mean=(),
    normalize_std=(),
    normalize: bool = True,
) -> transforms.Compose:
    steps = []
    if int(image_size) != 28:
        steps.append(transforms.Resize((int(image_size), int(image_size))))
    steps.append(transforms.ToTensor())
    if normalize and normalize_mean and normalize_std:
        steps.append(transforms.Normalize(normalize_mean, normalize_std))
    return transforms.Compose(steps)


def build_mnist_test_loader(
    config: DataConfig,
    image_size: Optional[int] = None,
    batch_size: int = 128,
    normalize: bool = True,
) -> DataLoader:
    if str(config.dataset).lower() != "mnist":
        raise NotImplementedError("Only MNIST is currently supported.")
    image_size = config.victim_resolution if image_size is None else int(image_size)
    test_set = datasets.MNIST(
        root=config.data_root,
        train=False,
        download=config.download,
        transform=_build_transform(image_size, config.normalize_mean, config.normalize_std, normalize=normalize),
    )
    test_set = _maybe_subset(test_set, int(config.test_subset))
    return DataLoader(
        test_set,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        drop_last=False,
    )


def load_mnist_class_image(
    config: DataConfig,
    target_label: int,
    image_size: Optional[int] = None,
    seed: int = 0,
    train: bool = True,
):
    if str(config.dataset).lower() != "mnist":
        raise NotImplementedError("Only MNIST is currently supported.")
    image_size = config.train_resolution if image_size is None else int(image_size)
    dataset = datasets.MNIST(
        root=config.data_root,
        train=bool(train),
        download=config.download,
        transform=_build_transform(image_size, config.normalize_mean, config.normalize_std, normalize=False),
    )
    targets = getattr(dataset, "targets")
    if hasattr(targets, "tolist"):
        targets = targets.tolist()
    indices = [idx for idx, label in enumerate(targets) if int(label) == int(target_label)]
    if not indices:
        split = "train" if train else "test"
        raise ValueError("No MNIST {} image found for target_label={}".format(split, target_label))
    image, _ = dataset[indices[int(seed) % len(indices)]]
    return image


def build_mnist_loaders(
    config: DataConfig,
    batch_size: int,
    train_image_size: Optional[int] = None,
    test_image_size: Optional[int] = None,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    if str(config.dataset).lower() != "mnist":
        raise NotImplementedError("Only MNIST is currently supported.")
    train_image_size = config.train_resolution if train_image_size is None else int(train_image_size)
    test_image_size = config.victim_resolution if test_image_size is None else int(test_image_size)
    train_set = datasets.MNIST(
        root=config.data_root,
        train=True,
        download=config.download,
        transform=_build_transform(train_image_size, config.normalize_mean, config.normalize_std, normalize=normalize),
    )
    train_set = _maybe_subset(train_set, int(config.train_subset))

    test_set = datasets.MNIST(
        root=config.data_root,
        train=False,
        download=config.download,
        transform=_build_transform(test_image_size, config.normalize_mean, config.normalize_std, normalize=normalize),
    )
    test_set = _maybe_subset(test_set, int(config.test_subset))

    train_loader = DataLoader(
        train_set,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        drop_last=False,
    )
    return train_loader, test_loader
