#include "network.h"

void network_layer_init(network_layer_t * layer, size_t weight_h, size_t weight_w, size_t bias_len, void (* activation)(float *, size_t, float *, size_t)) {
    assert(weight_h == bias_len);

    size_t weight_size = weight_h * weight_w;
    size_t buffer_size = weight_size + bias_len;

    layer->weights = (float *) malloc(buffer_size * sizeof(float));
    layer->weight_h = weight_h;
    layer->weight_w = weight_w;

    layer->bias = layer->weights + weight_size;
    layer->bias_len = bias_len;

    layer->activation = activation;
}

void network_layer_free(network_layer_t * layer) {
    free(layer->weights);
}


void network_init(network_t * network, size_t layer_len) {
    network->layers = (network_layer_t *) malloc(layer_len * sizeof(network_layer_t));
    network->layer_len = layer_len;
}

void network_free(network_t * network) {
    for(size_t i = 0; i < network->layer_len; ++i) {
	network_layer_free(network->layers + i);
    }
    free(network->layers);
}

void network_load_layers(network_t * network, FILE * data_file) {
    uint8_t buffer_2B [2];
    size_t num_read = fread(buffer_2B, sizeof(uint8_t), 2, data_file);
    assert(num_read == 2 && buffer_2B[0] == 0x50 && buffer_2B[1] == 0x54);

    size_t layer_idx = 0;
    uint32_t buffer_8B [2];
    while(!feof(data_file)) {
	num_read = fread(buffer_8B, sizeof(uint32_t), 2, data_file);
	if(num_read == 0 && feof(data_file)) break;
	assert(num_read > 0);
	size_t weight_h = (size_t)(ntohl(buffer_8B[0]));
	size_t weight_w = (size_t)(ntohl(buffer_8B[1]));
	size_t weight_size = weight_h * weight_w;

	assert(layer_idx < network->layer_len);
	network_layer_t * layer = network->layers + layer_idx;
	// invariant - bias_len is weight_h
	network_layer_init(layer, weight_h, weight_w, weight_h, imatmul_activation_relu);
	num_read = fread(layer->weights, sizeof(float), weight_size, data_file);
	assert(num_read == weight_size);

	num_read = fread(buffer_2B, sizeof(uint8_t), 1, data_file);
	assert(num_read == 1 && buffer_2B[0] == 0x00);

	num_read = fread(buffer_8B, sizeof(uint32_t), 2, data_file);
	assert((size_t)(ntohl(buffer_8B[0])) == weight_h);
	assert(ntohl(buffer_8B[1]) == 1);

	num_read = fread(layer->bias, sizeof(float), layer->bias_len, data_file);
	assert(num_read == layer->bias_len);

	num_read = fread(buffer_2B, sizeof(uint8_t), 1, data_file);
	assert(num_read == 1 && buffer_2B[0] == 0x00);

	// pack loaded params
	imatmul_params_t layer_params = {
	    /* params = */      layer->weights,
	    /* weight_h = */    layer->weight_h,
	    /* weight_w = */    layer->weight_w,
	    /* bias_offset = */ (layer->weight_h * layer->weight_w),
	    /* bias_length = */ layer->bias_len,
	    /* packed = */      0
	};
	imatmul_kernel_pack(&layer_params);
	assert(layer_params.packed);

	++layer_idx;
    }
    assert(layer_idx == network->layer_len);
}


void network_run(network_t * network, float * in_buffer, size_t in_buffer_len, float * out_buffer, size_t out_buffer_len) {
    assert(network->layer_len > 0);
    assert(network->layers[0].weight_w == in_buffer_len);
    size_t comp_buffer_len = network->layers[0].weight_h;
    float * comp_buffer = (float *) malloc(comp_buffer_len * sizeof(float));

    size_t in_len;
    size_t out_len;
    for(size_t i = 0; i < network->layer_len; ++i) {
	network_layer_t * layer = network->layers + i;
	imatmul_params_t matmul_params = {
	    /* params = */      layer->weights,
	    /* weight_h = */    layer->weight_h,
	    /* weight_w = */    layer->weight_w,
	    /* bias_offset = */ (layer->weight_h * layer->weight_w),
	    /* bias_length = */ layer->bias_len,
	    /* packed = */      1
	};
	in_len = layer->weight_w;
	out_len = layer->weight_h;
	assert(in_len <= in_buffer_len && out_len <= comp_buffer_len);

	imatmul_kernel_run(&matmul_params, in_buffer, in_len, comp_buffer, out_len);
	layer->activation(comp_buffer, out_len, in_buffer, out_len);
    }
    free(comp_buffer);

    assert(out_len == out_buffer_len);
    for(size_t i = 0; i < out_buffer_len; ++i) {
	out_buffer[i] = in_buffer[i];
    }
}
