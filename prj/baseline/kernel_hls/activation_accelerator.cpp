#include "activation_accelerator.h"
#include <iostream>
#include <cmath>
#include <hls_math.h>

/*
0 eltwise add
1 safe softmax tbd
2 mask safe softmax tbd
3 sigmoid
4 silu
5 rms norm
6 layer norm
*/







// bf16 bitwise addition function implementation
uint16 bf16add(uint16 a_bits, uint16 b_bits) {
    // Extract sign, exponent, and mantissa
    uint16_t a_sign = (a_bits >> 15) & 0x1;
    uint16_t b_sign = (b_bits >> 15) & 0x1;
    uint16_t a_exp = (a_bits >> 7) & 0xFF;
    uint16_t b_exp = (b_bits >> 7) & 0xFF;
    uint16_t a_mantissa = a_bits & 0x7F;
    uint16_t b_mantissa = b_bits & 0x7F;

    // Handle special case: zero
    if (a_exp == 0 && a_mantissa == 0) return b_bits;
    if (b_exp == 0 && b_mantissa == 0) return a_bits;

    // Increase precision for intermediate calculations (e.g., 8 extra bits for rounding)
    const int precision_shift = 8;

    // Construct full mantissa with implicit bit and increased precision
    uint32_t a_full_mantissa = (a_exp == 0) ? (a_mantissa) : ((0x80 | a_mantissa));
    uint32_t b_full_mantissa = (b_exp == 0) ? (b_mantissa) : ((0x80 | b_mantissa));

    a_full_mantissa <<= precision_shift;
    b_full_mantissa <<= precision_shift;

    // Adjust exponents for alignment (handle denormalized numbers)
    uint16_t a_align_exp = (a_exp == 0) ? 1 : a_exp;
    uint16_t b_align_exp = (b_exp == 0) ? 1 : b_exp;

    // Align exponents
    uint16_t max_exp;
    if (a_align_exp > b_align_exp) {
        max_exp = a_exp; // The result's exponent is based on the larger one
        b_full_mantissa >>= (a_align_exp - b_align_exp);
    } else if (b_align_exp > a_align_exp) {
        max_exp = b_exp;
        a_full_mantissa >>= (b_align_exp - a_align_exp);
    } else {
        max_exp = a_exp; // Both are equal
    }

    // Mantissa addition/subtraction
    uint32_t result_mantissa;
    uint16_t result_sign;
    if (a_sign == b_sign) {
        result_mantissa = a_full_mantissa + b_full_mantissa;
        result_sign = a_sign;
    } else {
        if (a_full_mantissa >= b_full_mantissa) {
            result_mantissa = a_full_mantissa - b_full_mantissa;
            result_sign = a_sign;
        } else {
            result_mantissa = b_full_mantissa - a_full_mantissa;
            result_sign = b_sign;
        }
    }

    // Return directly if the result is zero
    if (result_mantissa == 0) {
        return 0;
    }

    // Normalization
    // Representation of 1.0 is 0x80 << precision_shift
    // Representation of 2.0 is 0x100 << precision_shift
    while (result_mantissa < (0x80 << precision_shift) && max_exp > 0) {
        result_mantissa <<= 1;
        max_exp--;
        // If it becomes a denormalized number
        if (max_exp == 0) break;
    }

    if (result_mantissa >= (0x100 << precision_shift)) {
        result_mantissa >>= 1;
        max_exp++;
    }

    // Round half to nearest even
    uint32_t rounding_bits = result_mantissa & ((1 << precision_shift) - 1);
    uint32_t halfway = (1 << (precision_shift - 1));

    if (rounding_bits > halfway) {
        result_mantissa += (1 << precision_shift); // Round up
    } else if (rounding_bits == halfway) {
        // If it's a tie, round to nearest even
        if (((result_mantissa >> precision_shift) & 1) != 0) {
            result_mantissa += (1 << precision_shift); // Round up to make it even
        }
    }
    // else (rounding_bits < halfway), truncate (do nothing)


    // Renormalization might be needed after rounding
    if (result_mantissa >= (0x100 << precision_shift)) {
        result_mantissa >>= 1;
        max_exp++;
    }

    // Truncate back to original precision
    uint16_t final_mantissa_full = result_mantissa >> precision_shift;

    // Handle underflow to denormalized number
    if (max_exp == 0) {
        final_mantissa_full &= 0x7F;
    }

    // Remove the implicit bit
    uint16_t final_mantissa = final_mantissa_full & 0x7F;

    // Check for overflow
    if (max_exp >= 0xFF) {
        return (result_sign << 15) | (0xFF << 7); // Infinity
    }

    // Construct the final result
    return (result_sign << 15) | (max_exp << 7) | final_mantissa;
}




// bf16编码(uint16)批量转float32
void bf16_to_float(const uint16* in, float* out, int len) {
    for (int i = 0; i < len; ++i) {
        uint32_t x_f32 = ((uint32_t)in[i]) << 16;
        out[i] = *(float*)&x_f32;
    }
}
// sigmoid
void float_sigmoid(const float* x, uint16* y, int len) {
    for (int i = 0; i < len; ++i) {
        float val = 1.0f / (1.0f + hls::expf(-x[i]));
        uint32_t* y_f32_ptr = (uint32_t*)&val;
        y[i] = (*y_f32_ptr) >> 16;
    }
}
// silu
void float_silu(const float* x, uint16* y, int len) {
    for (int i = 0; i < len; ++i) {
        float sig = 1.0f / (1.0f + hls::expf(-x[i]));
        float val = x[i] * sig;
        uint32_t* y_f32_ptr = (uint32_t*)&val;
        y[i] = (*y_f32_ptr) >> 16;
    }
}
// rms_norm
void float_rms_norm(const float* x, uint16* y_bf16, int len) {
    const float eps = 1e-6f;
    float sum_sq = 0.0f;
    for (int i = 0; i < len; ++i) {
        sum_sq += x[i] * x[i];
    }
    float mean_sq = sum_sq / len;
    float rms = hls::sqrtf(mean_sq + eps);
    for (int i = 0; i < len; ++i) {
        float y = x[i] / rms;
        uint32_t* y_f32_ptr = (uint32_t*)&y;
        y_bf16[i] = (*y_f32_ptr) >> 16;
    }
}
// layer_norm
void float_layer_norm(const float* x, uint16* y_bf16, int len) {
    const float eps = 1e-6f;
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        sum += x[i];
    }
    float mean = sum / len;
    float var = 0.0f;
    for (int i = 0; i < len; ++i) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= len;
    float stddev = hls::sqrtf(var + eps);
    for (int i = 0; i < len; ++i) {
        float y = (x[i] - mean) / stddev;
        uint32_t* y_f32_ptr = (uint32_t*)&y;
        y_bf16[i] = (*y_f32_ptr) >> 16;
    }
}

// float加法
void float_add(const float* x, const float* y, uint16* out, int len) {
    for (int i = 0; i < len; ++i) {
        float sum = x[i] + y[i];
        uint32_t* sum_f32_ptr = (uint32_t*)&sum;
        out[i] = (*sum_f32_ptr) >> 16;
    }
}

// safe softmax
void float_safe_softmax(const float* x, uint16* y_bf16, int len) {
    float max_val = x[0];
    for (int i = 1; i < len; ++i) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    float exp_x[32768];
    for (int i = 0; i < len; ++i) {
        exp_x[i] = hls::expf(x[i] - max_val);
        sum += exp_x[i];
    }
    for (int i = 0; i < len; ++i) {
        float y = exp_x[i] / sum;
        uint32_t* y_f32_ptr = (uint32_t*)&y;
        y_bf16[i] = (*y_f32_ptr) >> 16;
    }
}
// mask safe softmax
void float_mask_safe_softmax(const float* x, const float* mask, uint16* y_bf16, int len) {
    float x_mask[32768];
    for (int i = 0; i < len; ++i) x_mask[i] = x[i] * mask[i];
    float max_val = x_mask[0];
    for (int i = 1; i < len; ++i) if (x_mask[i] > max_val) max_val = x_mask[i];
    float sum = 0.0f;
    float exp_x[32768];
    for (int i = 0; i < len; ++i) {
        exp_x[i] = hls::expf(x[i] - max_val);
        sum += exp_x[i];
    }
    for (int i = 0; i < len; ++i) {
        float y = exp_x[i] / sum;
        uint32_t* y_f32_ptr = (uint32_t*)&y;
        y_bf16[i] = (*y_f32_ptr) >> 16;
    }
}


void activation_accelerator(uint16* in0, uint16* in1, uint16* out, int32 stage, int32 config) {
#pragma HLS INTERFACE m_axi port=in0 offset=slave bundle=gmem0 depth=32768
#pragma HLS INTERFACE m_axi port=in1 offset=slave bundle=gmem1 depth=32768
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem2 depth=32768
#pragma HLS INTERFACE s_axilite port=stage
#pragma HLS INTERFACE s_axilite port=config
#pragma HLS INTERFACE s_axilite port=return

    static uint16 buf0[32*1024];
    static uint16 buf1[32*1024];
    static uint16 buf2[32*1024];
    float x[32*1024], y[32*1024];
    
    if(stage == 0) { // Stage 0: Load data from PS to PL
        for(int i = 0; i < 32*1024; i++) {
            buf0[i] = in0[i];
        }
        for(int i = 0; i < 32*1024; i++) {
            buf1[i] = in1[i];
        }
    }
    
    if(stage == 1) { // Stage 1: Compute
        if(config == 0) { // Element-wise addition
            bf16_to_float(buf0, x, 32*1024);
            bf16_to_float(buf1, y, 32*1024);
            float_add(x, y, buf2, 32*1024);
            for(int i = 0; i < 32*1024; i++) {
#pragma HLS PIPELINE II=1
                buf2[i] = bf16add(buf0[i], buf1[i]);
            }
        }
        else if(config == 1) { // safe softmax
            // bf16_to_float(buf0, x, 32*1024);
            // float_safe_softmax(x, buf2, 32*1024);
            for(int i = 0; i < 32*1024; i++) {
#pragma HLS PIPELINE II=1
                buf2[i] = 0;
            }
        }
        else if(config == 2) { // mask safe softmax
            // bf16_to_float(buf0, x, 32*1024);
            // bf16_to_float(buf1, y, 32*1024);
            // float_mask_safe_softmax(x, y, buf2, 32*1024);
            for(int i = 0; i < 32*1024; i++) {
#pragma HLS PIPELINE II=1
                buf2[i] = 0;
            }
        }
        else if(config == 3) { // Sigmoid
            bf16_to_float(buf0, x, 32*1024);
            float_sigmoid(x, buf2, 32*1024);
        }
        else if(config == 4) { // SiLU
            bf16_to_float(buf0, x, 32*1024);
            float_silu(x, buf2, 32*1024);
        }
        else if(config == 5) { // RMS normalization
            bf16_to_float(buf0, x, 32*1024);
            float_rms_norm(x, buf2, 32*1024);
        }
        else if(config == 6) { // Layer normalization
            bf16_to_float(buf0, x, 32*1024);
            float_layer_norm(x, buf2, 32*1024);
        }
    }
    
    if(stage == 2) { // Stage 2: Load data from PL to PS
        for(int i = 0; i < 32*1024; i++) {
            out[i] = buf2[i];
        }
    }
}
