def is_fp8_type(dtype):
    return dtype.is_floating_point and dtype.itemsize == 1


def is_8bit_type(dtype):
    return dtype.itemsize == 1
