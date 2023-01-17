import sys

sys.path.append("python")
import needle as ndl
import needle.nn as nn
import numpy as np
from quantize_weights import quantize_conv, dequantize_conv


def test_quant_conv():
    conv = nn.Conv(2, 3, 2, device=ndl.cpu())
    conv_weight = conv.weight.realize_cached_data()
    quantized_array, scales = quantize_conv(conv)
    deconv_weight = dequantize_conv(quantized_array, scales)
    differ = np.linalg.norm(conv_weight.numpy() - deconv_weight)

    print("original weights :", conv_weight)
    print("\n")
    print("quantize weights : ", quantized_array)
    print("shape: ", quantized_array.shape)
    print("scales: ", scales)
    print("\n")
    print("dequantized weights : ", deconv_weight)
    print("differs: ", differ)


def test_quant_model():
    # TODO
    pass


if __name__ == "__main__":

    test_quant_conv()
    test_quant_model()
