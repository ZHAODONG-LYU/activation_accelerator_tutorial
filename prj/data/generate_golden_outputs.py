#!/usr/bin/env python3
"""
Generate golden output data for all configurations
"""

import numpy as np
import torch
import struct
import os

def uint16_to_torch_bf16(data_uint16):
    """Convert uint16 bit pattern to torch.bfloat16"""
    # Parse bf16 format by bits: 1 sign bit + 8 exponent bits + 7 mantissa bits
    bf16_values = []
    for val in data_uint16:
        # Extract parts of bf16
        sign = (val >> 15) & 0x1
        exp = (val >> 7) & 0xFF
        mantissa = val & 0x7F
        
        # Special value handling
        if exp == 0:
            # Zero or subnormal number
            if mantissa == 0:
                result = 0.0
            else:
                # Subnormal number (rare in bf16)
                result = (1 if sign == 0 else -1) * (mantissa / 128.0) * (2 ** -126)
        elif exp == 255:
            # Infinity or NaN
            result = float('inf') if mantissa == 0 else float('nan')
            if sign == 1:
                result = -result
        else:
            # Normal number
            # bf16 bias is 127 (same as float32)
            exp_val = exp - 127.0
            mantissa_val = 1.0 + (mantissa / 128.0)
            result = (1.0 if sign == 0 else -1.0) * mantissa_val * (2.0 ** exp_val)
        
        # Directly use bf16 for calculation to avoid float32 precision loss
        bf16_val = torch.tensor(result, dtype=torch.bfloat16)
        bf16_values.append(bf16_val)
    
    return torch.stack(bf16_values)

def torch_bf16_to_uint16(tensor):
    """Convert torch.bfloat16 to uint16 bit pattern"""
    # Convert to float32, then get its bit pattern
    float32_values = tensor.float()
    uint32_values = np.array([struct.unpack('I', struct.pack('f', float(val)))[0] for val in float32_values], dtype=np.uint32)
    # Extract bit pattern of bfloat16 (sign bit + 8 exponent bits + 7 mantissa bits)
    # The trick here is to right shift 16 bits, discarding the lower 16 bits of float32 (i.e., the lower 16 bits of the mantissa)
    uint16_values = (uint32_values >> 16).astype(np.uint16)
    return uint16_values
def eltwise_add(x1, x2):
    """Element-wise addition"""
    result = x1 + x2
    
    # Output debug information only at error points
    python_result_uint16 = torch_bf16_to_uint16(result)
    
    # Check for errors (compare with previously generated golden data)
    try:
        golden_uint16 = np.fromfile(data_dir + 'golden_out_config_0_bf16.bin', dtype=np.uint16)
        error_count = 0
        for i in range(min(100, len(python_result_uint16))):
            if python_result_uint16[i] != golden_uint16[i]:
                error_count += 1
                if error_count <= 10:  # Only show the first 10 errors
                    print(f"Python Error[{i}]: expected={golden_uint16[i]}, actual={python_result_uint16[i]}")
                    print(f"    bf16add debug: a={torch_bf16_to_uint16(x1[i:i+1])[0]} "
                          f"(0x{torch_bf16_to_uint16(x1[i:i+1])[0]:04x}), "
                          f"b={torch_bf16_to_uint16(x2[i:i+1])[0]} "
                          f"(0x{torch_bf16_to_uint16(x2[i:i+1])[0]:04x}) "
                          f"-> result={python_result_uint16[i]} "
                          f"(0x{python_result_uint16[i]:04x})")
        if error_count > 0:
            print(f"Python total errors: {error_count} (in first 100 points)")
    except:
        pass  # If the golden file does not exist, do not output debug information
    
    return result

def safe_softmax(x, axis=-1):
    """Safe softmax"""
    max_ = torch.max(x, dim=axis, keepdim=True)[0]
    sub_ = x - max_
    exp_ = torch.exp(sub_)
    sum_ = torch.sum(exp_, dim=axis, keepdim=True)
    return exp_ / sum_

def mask_safe_softmax(x, mask, axis=-1):
    """Masked safe softmax"""
    x_mask = x * mask
    max_ = torch.max(x_mask, dim=axis, keepdim=True)[0]
    sub_ = x - max_
    exp_ = torch.exp(sub_)
    sum_ = torch.sum(exp_, dim=axis, keepdim=True)
    return exp_ / sum_

def sigmoid(x):
    """Sigmoid activation function"""
    x = torch.clamp(x, -500, 500)
    return 1 / (1 + torch.exp(-x))

def silu(x):
    """SiLU activation function"""
    return x * sigmoid(x)

def rms_norm(x, weight=None, eps=1e-6):
    """RMS normalization"""
    pow_ = x**2
    mean_ = torch.mean(pow_, dim=-1, keepdim=True)
    rms_ = torch.sqrt(mean_ + eps)
    x_norm = x / rms_
    if weight is not None:
        x_norm = x_norm * weight
    return x_norm

def layer_norm(x, weight=None, bias=None, eps=1e-6):
    """Layer normalization"""
    mean_ = torch.mean(x, dim=-1, keepdim=True)
    sub_ = x - mean_
    pow_ = torch.pow(sub_, 2)
    var_ = torch.mean(pow_, dim=-1, keepdim=True)
    sqrt_ = torch.sqrt(var_ + eps)
    norm_ = sub_ / sqrt_
    if weight is not None:
        norm_ = norm_ * weight
    if bias is not None:
        norm_ = norm_ + bias
    return norm_

def sigmoid_bf16(x_bf16):
    x_bf16 = x_bf16.to(torch.bfloat16)
    x_fp32 = x_bf16.float()
    y_fp32 = 1 / (1 + torch.exp(-x_fp32))
    return y_fp32.to(torch.bfloat16)

def generate_golden_outputs():
    """Generate golden outputs for all configurations"""
    print("Generating golden output data...")
    
    data_dir = "./"
    
    # Load input data
    print("Loading input data...")
    in0_uint16 = np.fromfile(data_dir + 'in0_bf16.bin', dtype=np.uint16)
    in1_uint16 = np.fromfile(data_dir + 'in1_bf16.bin', dtype=np.uint16)
    mask_uint16 = np.fromfile(data_dir + 'mask_bf16.bin', dtype=np.uint16)
    
    # Convert to torch.bfloat16
    in0 = uint16_to_torch_bf16(in0_uint16)
    in1 = uint16_to_torch_bf16(in1_uint16)
    mask = uint16_to_torch_bf16(mask_uint16)
    
    print(f"Data size: {len(in0)}")
    
    # Generate golden outputs for all configurations
    configs = list(range(7))
    
    for config in configs:
        if config == 0:
            # Element-wise addition
            output_data_bf16 = eltwise_add(in0, in1)
        elif config == 1:
            # Safe softmax
            # Here we assume the data is 1D, directly softmax
            output_data_bf16 = safe_softmax(in0.float()).to(torch.bfloat16)
        elif config == 2:
            # Masked softmax
            # Mask needs to be the same shape as in0
            output_data_bf16 = mask_safe_softmax(in0.float(), mask.float()).to(torch.bfloat16)
        elif config == 3:
            output_data_bf16 = sigmoid_bf16(in0)
        elif config == 4:
            # SiLU
            output_data_bf16 = silu(in0.float()).to(torch.bfloat16)
        elif config == 5:
            # RMS normalization
            output_data_bf16 = rms_norm(in0.float()).to(torch.bfloat16)
        elif config == 6:
            # Layer normalization
            output_data_bf16 = layer_norm(in0.float()).to(torch.bfloat16)
        else:
            output_data_bf16 = torch.zeros_like(in0)
        
        # Ensure the result is 1D
        if output_data_bf16.dim() > 1:
            output_data_bf16 = output_data_bf16.flatten()
        
        # Truncate or pad to the correct length
        if len(output_data_bf16) > len(in0):
            output_data_bf16 = output_data_bf16[:len(in0)]
        elif len(output_data_bf16) < len(in0):
            padding = torch.zeros(len(in0) - len(output_data_bf16), dtype=torch.bfloat16)
            output_data_bf16 = torch.cat([output_data_bf16, padding])
        
        # Convert to uint16 and save
        result_uint16 = torch_bf16_to_uint16(output_data_bf16)
        output_file = data_dir + f'golden_out_config_{config}_bf16.bin'
        result_uint16.tofile(output_file)
        
        print(f"  Saved to: {output_file}")
        print(f"  Result stats: min={float(output_data_bf16.min()):.6f}, max={float(output_data_bf16.max()):.6f}")
    
    print("Golden output generation complete!")

if __name__ == "__main__":
    generate_golden_outputs()
