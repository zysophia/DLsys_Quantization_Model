import sys
sys.path.append('./python')
import numpy as np
import needle as ndl
import needle.nn as nn

from typing import Tuple


def dequantize_conv(data, scales):
    return (data * scales.reshape(1,1,1,-1)).astype(np.float32)


def symmetric_quantize_weight(arr: np.ndarray) -> Tuple[np.ndarray, float]:
     data_abs_max = max(abs(arr.min()), abs(arr.max()))
     scale = data_abs_max / 127
     quantized_arr = np.round((1 / scale) * arr)
     quantized_arr = np.clip(quantized_arr, a_min=-128, a_max=127).astype(np.int8)
     return quantized_arr, scale
  

def quantize_conv(x: nn.Conv) -> Tuple[np.ndarray, np.ndarray]:
    data = x.weight.realize_cached_data().numpy() # convert to numpy for simplicity
    channels = data.shape[3]
    quantized_data = np.empty_like(data, dtype=np.int8)
    scales = np.empty(channels)

    for c in range(channels): # quantize by channel
        quantized_data[:,:,:,c], scales[c] = symmetric_quantize_weight(data[:,:,:,c])
  
    return quantized_data, scales # TODO: not sure where to save quantized data and scales
