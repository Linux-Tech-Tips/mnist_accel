#include <stdio.h>

#include "network.h"

#define IMAGE_SIZE 28*28
#define OUTPUT_SIZE 10

void load_test_img(float image [IMAGE_SIZE], FILE * img_file);

uint8_t argmax(float * values, size_t len);

int main(int argc, char * argv[]) {
    assert(argc > 2);

    const char * network_name = argv[1];
    FILE * network_file = fopen(network_name, "r");

    network_t net = {0};
    network_init(&net, 3);
    network_load_layers(&net, network_file);

    puts("network loading complete!");

    const char * img_name = argv[2];
    FILE * img_file = fopen(img_name, "r");

    float image [IMAGE_SIZE];
    load_test_img(image, img_file);

    puts("image loading complete!");

    float out [OUTPUT_SIZE];
    network_run(&net, image, IMAGE_SIZE, out, OUTPUT_SIZE);
    uint8_t result = argmax(out, OUTPUT_SIZE);
    printf("output: %u\n", result);

    network_free(&net);

    return 0;
}

void load_test_img(float image [IMAGE_SIZE], FILE * img_file) {
    uint8_t buffer_2B [2];
    size_t num_read = fread(buffer_2B, sizeof(uint8_t), 2, img_file);
    assert(num_read == 2 && buffer_2B[0] == 0x49 && buffer_2B[1] == 0x4D);

    num_read = fread(image, sizeof(float), IMAGE_SIZE, img_file);
    assert(num_read == IMAGE_SIZE);
}

uint8_t argmax(float * values, size_t len) {
    assert(len > 0);
    float max = values[0];
    uint8_t idx = 0;
    for(size_t i = 1; i < len; ++i) {
	if(values[i] > max) {
	    max = values[i];
	    idx = i;
	}
    }
    return idx;
}
