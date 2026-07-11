#include <stdio.h>

#include "network.h"

#define IMAGE_SIZE 28*28

void load_test_img(float image [IMAGE_SIZE], FILE * img_file);

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
