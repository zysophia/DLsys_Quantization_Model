import sys

sys.path.append("python")
import needle as ndl
import needle.nn as nn
import numpy as np
from quantize import dequantize_conv, quantize_conv


def test_quant_conv():
    inp = ndl.Tensor(np.random.randn(1, 3, 32, 32), device=ndl.cpu())

    conv = nn.Conv(3, 8, 3, device=ndl.cpu())
    conv_weight = conv.weight.realize_cached_data()
    quantized_array, scales = quantize_conv(conv)
    deconv_weight = dequantize_conv(quantized_array, scales)
    differ = np.linalg.norm(conv_weight.numpy() - deconv_weight)

    print("quant/dequant differs: ", differ)
    qconv = nn.QConv(quantized_array, scales, bias=conv.bias, device=ndl.cpu())

    fp_out = conv(inp)
    q_out = qconv(inp)

    differ = np.linalg.norm(fp_out.numpy() - q_out.numpy())
    print("Infer differs: ", differ)


def test_quant_model():
    # TODO
    pass


if __name__ == "__main__":

    test_quant_conv()
    test_quant_model()
