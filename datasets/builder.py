from __future__ import annotations

from utils.mnist import build_mnist_loaders, build_mnist_test_loader


def build_train_and_test_loaders(
    config,
    *,
    batch_size=None,
    train_image_size=None,
    test_image_size=None,
    normalize: bool = True,
):
    data_config = config.data if hasattr(config, "data") else config
    train_config = getattr(config, "train", None)
    resolved_batch_size = batch_size if batch_size is not None else getattr(train_config, "batch_size", 128)
    return build_mnist_loaders(
        data_config,
        batch_size=int(resolved_batch_size),
        train_image_size=int(train_image_size if train_image_size is not None else data_config.train_resolution),
        test_image_size=int(test_image_size if test_image_size is not None else data_config.victim_resolution),
        normalize=bool(normalize),
    )


def build_test_loader(
    config,
    *,
    image_size=None,
    batch_size=None,
    normalize: bool = True,
):
    data_config = config.data if hasattr(config, "data") else config
    train_config = getattr(config, "train", None)
    resolved_batch_size = batch_size if batch_size is not None else getattr(train_config, "batch_size", 128)
    return build_mnist_test_loader(
        data_config,
        image_size=int(image_size if image_size is not None else data_config.victim_resolution),
        batch_size=int(resolved_batch_size),
        normalize=bool(normalize),
    )


def build_train_loader_pair(
    config,
    *,
    batch_size=None,
    train_image_size=None,
    test_image_size=None,
    normalize: bool = True,
):
    train_loader1, _ = build_train_and_test_loaders(
        config,
        batch_size=batch_size,
        train_image_size=train_image_size,
        test_image_size=test_image_size,
        normalize=normalize,
    )
    train_loader2, _ = build_train_and_test_loaders(
        config,
        batch_size=batch_size,
        train_image_size=train_image_size,
        test_image_size=test_image_size,
        normalize=normalize,
    )
    return train_loader1, train_loader2
