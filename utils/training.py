def set_requires_grad(module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def count_parameters(params) -> int:
    return sum(p.numel() for p in params if p.requires_grad)
