import sys
sys.path.append('python')
sys.path.append('tools')
import needle as ndl
import needle.nn as nn
from quantize_weights import quantize_conv, dequantize_conv

conv = nn.Conv(2,3,2, device=ndl.cpu())

print('original weights :', conv.weight.realize_cached_data())
print('\n')
quantized_array, scales = quantize_conv(conv)
print('quantize weights : ', quantized_array)
print('shape: ', quantized_array.shape)
print('\n')
print('scales: ' , scales)
print('\n')
print('dequantized weights : ', dequantize_conv(quantized_array, scales))