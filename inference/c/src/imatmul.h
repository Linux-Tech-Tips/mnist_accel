/** 
 * @file imatmul.h
 * @author Linux-Tech-Tips (Martin)
 * @brief interface function definitions for accelerated matrix-vector multiplication kernel
 */
#ifndef I_MATMUL_H
#define I_MATMUL_H

#include <stddef.h>
#include <stdint.h>

/** Parameters for matrix multiplication, used by the interface functions for inference */
typedef struct {
    /** pointer to the allocated memory space for the weights and bias, contiguous, expects weights in row-major format */
    float * params;
    /** the height of the weight matrix (number of rows) */
    size_t weight_h;
    /** the width of the weight matrix (number of columns) */
    size_t weight_w;

    /** the offset at which the bias vector starts (weights + bias_offset is a pointer to the bias vector) */
    size_t bias_offset;
    /** the length of the bias vector */
    size_t bias_length;

    /** whether the params have been packed */
    uint8_t packed;
} imatmul_params_t;

/** Run an in-place pre-pack in RAM for weights and bias of a given layer */
void imatmul_kernel_pack(imatmul_params_t * params);

/** Run a matrix-vector multiplication given the provided weights, bias (params) and input buffer 
 * @param params weights matrix and bias for the matrix-vector multiplication
 * @param in_buffer memory buffer containing the input vector data
 * @param in_buffer_len length of the input vector
 * @param out_buffer memory buffer to hold the output vector data
 * @param out_buffer_len expected length of the output vector
 */
void imatmul_kernel_run(imatmul_params_t * params, float * in_buffer, size_t in_buffer_len, float * out_buffer, size_t out_buffer_len);

/** Run a ReLU activation function on the given in_buffer and save output to out_buffer */
void imatmul_activation_relu(float * in_buffer, size_t in_buffer_len, float * out_buffer, size_t out_buffer_len);

#endif /* I_MATMUL_H */
