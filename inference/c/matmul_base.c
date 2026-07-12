#include "imatmul.h"

#include <assert.h>

void imatmul_kernel_pack(imatmul_params_t * params) {
    // no repacking required for base matmul
    params->packed = 1;
}

void imatmul_kernel_run(imatmul_params_t * params, float * in_buffer, size_t in_buffer_len, float * out_buffer, size_t out_buffer_len) {
    assert(out_buffer_len == params->weight_h);
    assert(out_buffer_len == params->bias_length);
    for(size_t m = 0; m < out_buffer_len; ++m) {
	out_buffer[m] = params->params[params->bias_offset+m];
	for(size_t k = 0; k < params->weight_w; ++k) {
	    out_buffer[m] += params->params[(m*params->weight_w+k)] * in_buffer[k];
	}
    }
}

void imatmul_activation_relu(float * in_buffer, size_t in_buffer_len, float * out_buffer, size_t out_buffer_len) {
    assert(in_buffer_len == out_buffer_len);
    for(size_t i = 0; i < out_buffer_len; ++i) out_buffer[i] = (in_buffer[i] > 0 ? in_buffer[i] : 0);
}
