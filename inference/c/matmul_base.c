#include "imatmul.h"

void imatmul_kernel_pack(imatmul_params_t * params) {
    // no repacking required for base matmul
    params->packed = 1;
}

void imatmul_kernel_run(imatmul_params_t * params, float * in_buffer, size_t in_buffer_len, float * out_buffer, size_t out_buffer_len) {
    // TODO Implement
    return;
}

void imatmul_activation_relu(float * in_buffer, size_t in_buffer_len, float * out_buffer, size_t out_buffer_len) {
    // TODO Implement
    return;
}
