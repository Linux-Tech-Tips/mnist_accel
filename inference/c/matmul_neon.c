#include "matmul_neon.h"

/** Arm NEON SIMD matrix-vector multiplication implementation
 *
 * - Packing approach:
 *   - for optimal contiguous memory access
 *   - block column-major format
 *   - block of height K from top of first column, then second, until end of params
 *     followed by another block of height K from top offset by K
 *   - K is the row size, where R is the SIMD register length and K % R == 0 and K/R defines how much to unroll the matmul
 */

/** Number of F32 elements in single register */
#define R 4
/** Number of usable Q (quad-word) registers for 1 vector input and 1 vector output VMLA (scalar) op */
#define Q 8
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
    // TODO Matmul Kernel Run using ACLE on packed data

    // TODO What do here:
    //  -> set up Q NEON Q vector registers - out
    //  -> set up Q NEON Q vector registers - in
    //  -> iterate over params->weight_h (check it's same as out_buffer_len) in steps of K
    //  -> for each iteration:
    //    -> mem load K elements from params into out registers in steps of R (Q steps unrolled)
    //    -> iterate over params->weight_w, for each:
    //      -> mem load K elements from params into in registers in steps of R (Q steps unrolled)
    //      -> VMLA (scalar) in:in_r in_scalar:weight_w_offset out:out_r (Q times unrolled)
    //    -> mem write K elements from out registers into out_buffer (Q steps unrolled)

    assert(Q == 8); // Unwrapped for Q == 8

    float32x4_t out1;
    float32x4_t out2;
    float32x4_t out3;
    float32x4_t out4;
    float32x4_t out5;
    float32x4_t out6;
    float32x4_t out7;
    float32x4_t out8;

    float32x4_t in1;
    float32x4_t in2;
    float32x4_t in3;
    float32x4_t in4;
    float32x4_t in5;
    float32x4_t in6;
    float32x4_t in7;
    float32x4_t in8;

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
	out5 = vld1q_f32(params_ptr);
	params_ptr += R;
	out6 = vld1q_f32(params_ptr);
	params_ptr += R;
	out7 = vld1q_f32(params_ptr);
	params_ptr += R;
	out8 = vld1q_f32(params_ptr);
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
	    in5 = vld1q_f32(params_ptr);
	    params_ptr += R;
	    in6 = vld1q_f32(params_ptr);
	    params_ptr += R;
	    in7 = vld1q_f32(params_ptr);
	    params_ptr += R;
	    in8 = vld1q_f32(params_ptr);
	    params_ptr += R;

	    float c = in_buffer[w_offset];
	    out1 = vmlaq_n_f32(out1, in1, c);
	    out2 = vmlaq_n_f32(out2, in2, c);
	    out3 = vmlaq_n_f32(out3, in3, c);
	    out4 = vmlaq_n_f32(out4, in4, c);
	    out5 = vmlaq_n_f32(out5, in5, c);
	    out6 = vmlaq_n_f32(out6, in6, c);
	    out7 = vmlaq_n_f32(out7, in7, c);
	    out8 = vmlaq_n_f32(out8, in8, c);

	}

	vst1q_f32(out_buffer + out_buffer_idx, out1);
        out_buffer_idx += R;
	vst1q_f32(out_buffer + out_buffer_idx, out2);
        out_buffer_idx += R;
	vst1q_f32(out_buffer + out_buffer_idx, out3);
        out_buffer_idx += R;
	vst1q_f32(out_buffer + out_buffer_idx, out4);
        out_buffer_idx += R;
	vst1q_f32(out_buffer + out_buffer_idx, out5);
        out_buffer_idx += R;
	vst1q_f32(out_buffer + out_buffer_idx, out6);
        out_buffer_idx += R;
	vst1q_f32(out_buffer + out_buffer_idx, out7);
        out_buffer_idx += R;
	vst1q_f32(out_buffer + out_buffer_idx, out8);
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
    // TODO What here:
    // ReLU abs, add then vector multiply by scalar 0.5f
    // -> iterate over in_buffer_len in steps of K
    //   -> load K elements into Q in-registers and Q out-registers
    //   -> VABS on Q in-registers in-place
    //   -> VADD in-registers and out-registers into out-registers
    //   -> VMUL by scalar constant 0.5f out-registers
    //   -> save K elements from Q out-registers into out_buffer

    float32x4_t out1;
    float32x4_t out2;
    float32x4_t out3;
    float32x4_t out4;
    float32x4_t out5;
    float32x4_t out6;
    float32x4_t out7;
    float32x4_t out8;

    float32x4_t in1;
    float32x4_t in2;
    float32x4_t in3;
    float32x4_t in4;
    float32x4_t in5;
    float32x4_t in6;
    float32x4_t in7;
    float32x4_t in8;

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
	out5 = vld1q_f32(in_buffer);
	in5 = vld1q_f32(in_buffer);
	in_buffer += R;
	out6 = vld1q_f32(in_buffer);
	in6 = vld1q_f32(in_buffer);
	in_buffer += R;
	out7 = vld1q_f32(in_buffer);
	in7 = vld1q_f32(in_buffer);
	in_buffer += R;
	out8 = vld1q_f32(in_buffer);
	in8 = vld1q_f32(in_buffer);
	in_buffer += R;

	in1 = vabsq_f32(in1);
	in2 = vabsq_f32(in2);
	in3 = vabsq_f32(in3);
	in4 = vabsq_f32(in4);
	in5 = vabsq_f32(in5);
	in6 = vabsq_f32(in6);
	in7 = vabsq_f32(in7);
	in8 = vabsq_f32(in8);

	out1 = vaddq_f32(in1, out1);
	out2 = vaddq_f32(in2, out2);
	out3 = vaddq_f32(in3, out3);
	out4 = vaddq_f32(in4, out4);
	out5 = vaddq_f32(in5, out5);
	out6 = vaddq_f32(in6, out6);
	out7 = vaddq_f32(in7, out7);
	out8 = vaddq_f32(in8, out8);

	const float c = 0.5f;
	out1 = vmulq_n_f32(out1, c);
	out2 = vmulq_n_f32(out2, c);
	out3 = vmulq_n_f32(out3, c);
	out4 = vmulq_n_f32(out4, c);
	out5 = vmulq_n_f32(out5, c);
	out6 = vmulq_n_f32(out6, c);
	out7 = vmulq_n_f32(out7, c);
	out8 = vmulq_n_f32(out8, c);

	vst1q_f32(out_buffer, out1);
	out_buffer += R;
	vst1q_f32(out_buffer, out2);
	out_buffer += R;
	vst1q_f32(out_buffer, out3);
	out_buffer += R;
	vst1q_f32(out_buffer, out4);
	out_buffer += R;
	vst1q_f32(out_buffer, out5);
	out_buffer += R;
	vst1q_f32(out_buffer, out6);
	out_buffer += R;
	vst1q_f32(out_buffer, out7);
	out_buffer += R;
	vst1q_f32(out_buffer, out8);
	out_buffer += R;

    }

    // tail
    for(; offset < in_buffer_len; ++offset) {
	float in = *(in_buffer++);
	*(out_buffer++) = (in + (in > 0 ? in : -in)) / 2;
    }
}
