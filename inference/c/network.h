#ifndef NETWORK_H
#define NETWORK_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <arpa/inet.h>

#include "imatmul.h"

typedef struct {
    float * weights;
    size_t weight_h;
    size_t weight_w;

    float * bias;
    size_t bias_len;

    void (* activation)(float *, size_t, float *, size_t);
} network_layer_t;

typedef struct {
    network_layer_t * layers;
    size_t layer_len;
} network_t;


void network_layer_init(network_layer_t * layer, size_t weight_h, size_t weight_w, size_t bias_len, void (* activation)(float *, size_t, float *, size_t));

void network_layer_free(network_layer_t * layer);


void network_init(network_t * network, size_t layer_len);

void network_free(network_t * network);

void network_load_layers(network_t * network, FILE * data_file);


void network_run(network_t * network, float * in_buffer, size_t in_buffer_len, float * out_buffer, size_t out_buffer_len);

#endif /* NETWORK_H */
