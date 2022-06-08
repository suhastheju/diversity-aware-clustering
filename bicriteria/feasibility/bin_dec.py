from math import floor

import numpy as np


def bin_to_int(bit_vector):
    int_value = 0
    for bit in bit_vector:
        int_value = (int_value << 1) | bit
    # end for
    return int_value


def int_to_bin(int_value, length):
    bit_vector = np.zeros(length, dtype=np.uint8)
    bit_index = 0
    while (int_value > 0):
        bit = int_value & 0x01
        bit_vector[bit_index] = bit
        int_value = int_value >> 1
        bit_index += 1
    # end while
    return bit_vector


def k_array_to_int(a, num_of_medians):
    m = 1
    result = 0
    for i in range(a.size, 0, -1):
        result = result + a[i - 1] * m
        m = m * (num_of_medians + 1)
    return result


def int_to_k_array(a, num_of_medians, num_colors):
    result = np.zeros(num_colors, dtype=np.int8)
    i = num_colors - 1
    while (a != 0):
        result[i] = a % (num_of_medians + 1)
        a = floor(a / (num_of_medians + 1))
        i = i - 1
    # end while
    return result
