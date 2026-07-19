#include "matmul_neon.h"

/** Arm NEON SIMD matrix-vector multiplication implementation
 *
 * - Packing approach:
 *   - for optimal contiguous (spacial locality cache-friendly) memory access
 *   - block column-major format
 *   - block of height K from top of first column, then second, until end of params
 *     followed by another block of height K from top offset by K
 *   - K is the row size, where R is the SIMD register length and K % R == 0 and K/R defines how much to unroll the matmul
 *
 * - Matrix-Vector Multiplication approach:
 *   - utilising memory pre-packed layout for efficient contiguous loads
 *   - parallel weight*input calculation using Arm64 VMLA vector-multiply by scalar instruction
 *   - loop logic:
 *     - iterate over weight height in steps of K
 *     - mem load next K elements from params into out registers in steps of R (Q steps unrolled) - bias
 *     - iterate over weight width, load next K elements from params - next K weights
 *     - VMLA vector-multiply by scalar - current K weights by current width index input, accumulate into K out registers
 *     - write current K out registers into output buffer
 *
 * - ReLU Actiation approach:
 *   - using formula `ReLU(x) = (abs(x) + x) * 0.5f`
 *   - iterating over in-buffer length in steps of K:
 *     - load K elements into Q in-registers and Q out-registers (2Q unrolled steps)
 *     - VABS the Q in-registers in-place
 *     - VADD in-registers and out-registers into out-registers
 *     - VMUL by scalar constant 0.5f
 *     - save K elements from Q out-registers into next K elements in out-buffer
 *
 */

/** Number of F32 elements in single register */
#define R 4
/** Number of used Q (quad-word) registers for 1 vector input and 1 vector output VMLA (scalar) op */
#define Q 4
/** Step in which to go down the weight matrix */
#define K (R*Q)

void imatmul_kernel_pack(imatmul_params_t * params) {
    size_t params_total = params->weight_h * params->weight_w + params->bias_length;
    float * old_params = (float *) malloc(params_total * sizeof(float));
    memcpy(old_params, params->params, params_total * sizeof(float));

    size_t params_idx = 0;
    size_t h_offset = 0;
    for(; (params->weight_h >= K) && h_offset <= (params->weight_h - K) && params_idx < params_total; h_offset += K) {
	// bias
	memcpy((params->params + params_idx), (old_params + params->bias_offset + h_offset), K * sizeof(float));
	params_idx += K;

	// weight block
	for(size_t w_offset = 0; w_offset < params->weight_w; ++w_offset) {
	    for(size_t k_offset = 0; k_offset < K; ++k_offset) {
		size_t old_params_offset = (h_offset + k_offset) * params->weight_w + w_offset;
		params->params[params_idx++] = old_params[old_params_offset];
	    }
	}
    }
    // tail
    if(h_offset < params->weight_h && params_idx < params_total) {
	size_t tail_k = params->weight_h - h_offset;
	//bias
	memcpy((params->params + params_idx), (old_params + params->bias_offset + h_offset), tail_k * sizeof(float));
	params_idx += tail_k;

	// weight block
	for(size_t w_offset = 0; w_offset < params->weight_w; ++w_offset) {
	    for(size_t k_offset = 0; k_offset < tail_k; ++k_offset) {
		size_t old_params_offset = (h_offset + k_offset) * params->weight_w + w_offset;
		params->params[params_idx++] = old_params[old_params_offset];
	    }
	}
    }

    free(old_params);
    assert(params_idx == params_total);
    params->packed = 1;
}

void imatmul_kernel_run(imatmul_params_t * params, float * in_buffer, size_t in_buffer_len, float * out_buffer, size_t out_buffer_len) {

    assert(Q == 4); // Unwrapped for Q == 4

    float32x4_t out1;
    float32x4_t out2;
    float32x4_t out3;
    float32x4_t out4;

    float32x4_t in1;
    float32x4_t in2;
    float32x4_t in3;
    float32x4_t in4;

    assert(params->weight_h == out_buffer_len);
    assert(params->weight_w == in_buffer_len);
    float * params_ptr = params->params;
    size_t h_offset = 0;
    size_t out_buffer_idx = 0;
    for(; (params->weight_h >= K) && h_offset <= (params->weight_h - K); h_offset += K) {

	out1 = vld1q_f32(params_ptr);
	params_ptr += R;
	out2 = vld1q_f32(params_ptr);
	params_ptr += R;
	out3 = vld1q_f32(params_ptr);
	params_ptr += R;
	out4 = vld1q_f32(params_ptr);
	params_ptr += R;

	for(size_t w_offset = 0; w_offset < params->weight_w; ++w_offset) {

	    in1 = vld1q_f32(params_ptr);
	    params_ptr += R;
	    in2 = vld1q_f32(params_ptr);
	    params_ptr += R;
	    in3 = vld1q_f32(params_ptr);
	    params_ptr += R;
	    in4 = vld1q_f32(params_ptr);
	    params_ptr += R;

	    float c = in_buffer[w_offset];
	    out1 = vmlaq_n_f32(out1, in1, c);
	    out2 = vmlaq_n_f32(out2, in2, c);
	    out3 = vmlaq_n_f32(out3, in3, c);
	    out4 = vmlaq_n_f32(out4, in4, c);

	}

	vst1q_f32(out_buffer + out_buffer_idx, out1);
        out_buffer_idx += R;
	vst1q_f32(out_buffer + out_buffer_idx, out2);
        out_buffer_idx += R;
	vst1q_f32(out_buffer + out_buffer_idx, out3);
        out_buffer_idx += R;
	vst1q_f32(out_buffer + out_buffer_idx, out4);
        out_buffer_idx += R;
    }

    // tail
    if(h_offset < params->weight_h) {
	size_t tail_h = params->weight_h - h_offset;
	memcpy((out_buffer + out_buffer_idx), params_ptr, tail_h * sizeof(float));
        params_ptr += tail_h;

	for(size_t w_offset = 0; w_offset < params->weight_w; ++w_offset) {
	    for(size_t i = 0; i < tail_h; ++i) {
		out_buffer[out_buffer_idx + i] += *(params_ptr++) * in_buffer[w_offset];
	    }
	}
    }
}

void imatmul_activation_relu(float * in_buffer, size_t in_buffer_len, float * out_buffer, size_t out_buffer_len) {

    float32x4_t out1;
    float32x4_t out2;
    float32x4_t out3;
    float32x4_t out4;

    float32x4_t in1;
    float32x4_t in2;
    float32x4_t in3;
    float32x4_t in4;

    size_t offset = 0;
    for(; (in_buffer_len >= K) && offset <= (in_buffer_len - K); offset += K) {

	out1 = vld1q_f32(in_buffer);
	in1 = vld1q_f32(in_buffer);
	in_buffer += R;
	out2 = vld1q_f32(in_buffer);
	in2 = vld1q_f32(in_buffer);
	in_buffer += R;
	out3 = vld1q_f32(in_buffer);
	in3 = vld1q_f32(in_buffer);
	in_buffer += R;
	out4 = vld1q_f32(in_buffer);
	in4 = vld1q_f32(in_buffer);
	in_buffer += R;

	in1 = vabsq_f32(in1);
	in2 = vabsq_f32(in2);
	in3 = vabsq_f32(in3);
	in4 = vabsq_f32(in4);

	out1 = vaddq_f32(in1, out1);
	out2 = vaddq_f32(in2, out2);
	out3 = vaddq_f32(in3, out3);
	out4 = vaddq_f32(in4, out4);

	const float c = 0.5f;
	out1 = vmulq_n_f32(out1, c);
	out2 = vmulq_n_f32(out2, c);
	out3 = vmulq_n_f32(out3, c);
	out4 = vmulq_n_f32(out4, c);

	vst1q_f32(out_buffer, out1);
	out_buffer += R;
	vst1q_f32(out_buffer, out2);
	out_buffer += R;
	vst1q_f32(out_buffer, out3);
	out_buffer += R;
	vst1q_f32(out_buffer, out4);
	out_buffer += R;

    }

    // tail
    for(; offset < in_buffer_len; ++offset) {
	float in = *(in_buffer++);
	*(out_buffer++) = (in + (in > 0 ? in : -in)) / 2;
    }
}
