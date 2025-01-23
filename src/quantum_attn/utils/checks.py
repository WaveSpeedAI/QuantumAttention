import functools

import torch
from packaging import version


def torch_version_compare(op, v):
    return getattr(version.parse(torch.__version__).release, f"__{op}__")(version.parse(v).release)


@functools.cache
def has_triton_package() -> bool:
    try:
        import triton

        return triton is not None
    except ImportError:
        return False


def triton_version_compare(op, v):
    if not has_triton_package():
        return None
    import triton

    return getattr(version.parse(triton.__version__).release, f"__{op}__")(version.parse(v).release)


def has_triton_language(attr):
    if not has_triton_package():
        return False
    import triton.language as tl

    return hasattr(tl, attr)


def has_triton_tma_support():
    if not has_triton_language("_experimental_descriptor_load"):
        return False

    import triton.language as tl

    return hasattr(tl.extra.cuda, "experimental_tensormap_fenceproxy_acquire")


def is_nvidia_cuda():
    return torch.version.hip is None and torch.cuda.is_available()


def cuda_capability_compare(op, major, minor, *, device=None):
    if not is_nvidia_cuda():
        return None
    return getattr(torch.cuda.get_device_capability(device), f"__{op}__")((major, minor))


def torch_cuda_version():
    if torch.version.cuda is None:
        return (0, 0)
    cuda_version = str(torch.version.cuda)
    return tuple(int(x) for x in cuda_version.split("."))[:2]


def torch_cuda_version_compare(op, major, minor):
    return getattr(torch_cuda_version(), f"__{op}__")((major, minor))
