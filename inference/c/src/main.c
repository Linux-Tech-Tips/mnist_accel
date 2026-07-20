#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "network.h"

#define IMAGE_SIZE 28*28
#define OUTPUT_SIZE 10

void load_test_img(float image [IMAGE_SIZE], FILE * img_file);

uint8_t argmax(float * values, size_t len);

uint64_t get_time_us();

int main(int argc, char * argv[]) {
    assert(argc > 0);

    if(argc < 2) {
        printf("%s needs first argument 'run' or 'bench'\n", argv[0]);
        return 1;
    }

    if(argc < 3) {
        printf("%s needs second argument model dat file name\n", argv[0]);
        return 1;
    }

    const char * network_name = argv[2];
    FILE * network_file = fopen(network_name, "r");
    network_t net = {0};

    if(strcmp(argv[1], "run") == 0) {
        if(argc < 4) {
            puts("'run' needs last argument image dat file name");
            fclose(network_file);
            return 1;
        }

        network_init(&net, 3);
        network_load_layers(&net, network_file);

        const char * img_name = argv[3];
        FILE * img_file = fopen(img_name, "r");
        float image [IMAGE_SIZE];
        load_test_img(image, img_file);
        fclose(img_file);

        float out [OUTPUT_SIZE];
        network_run(&net, image, IMAGE_SIZE, out, OUTPUT_SIZE);
        uint8_t result = argmax(out, OUTPUT_SIZE);
        printf("output: %u\n", result);

    } else if(strcmp(argv[1], "bench") == 0) {
        if(argc < 5) {
            puts("'bench' needs repeat count and at least one image arguments");
            fclose(network_file);
            return 1;
        }

        uint64_t model_load_start = get_time_us();
        network_t net = {0};
        network_init(&net, 3);
        network_load_layers(&net, network_file);
        uint64_t model_load_end = get_time_us();

        size_t repeats = (size_t) strtol(argv[3], NULL, 10);
        if(repeats == 0) {
            puts("'bench' needs non-zero repeat value");
            fclose(network_file);
            network_free(&net);
            return 1;
        }

        size_t images_len = argc - 4;
        float ** images = (float **) malloc(images_len * sizeof(float *));
        for(size_t i = 0; i < images_len; ++i) {
            images[i] = (float *) malloc(IMAGE_SIZE * sizeof(float));
            FILE * img_file = fopen(argv[i + 4], "r");
            load_test_img(images[i], img_file);
            fclose(img_file);
        }

        uint64_t run_total = 0;
        float out [OUTPUT_SIZE];
        for(size_t i = 0; i < repeats; ++i) {
            for(size_t img_idx = 0; img_idx < images_len; ++img_idx) {
                uint64_t run_start = get_time_us();
                network_run(&net, images[img_idx], IMAGE_SIZE, out, OUTPUT_SIZE);
                argmax(out, OUTPUT_SIZE);
                uint64_t run_end = get_time_us();
                run_total += (run_end - run_start);
            }
        }

        for(size_t i = 0; i < images_len; ++i) {
            free(images[i]);
        }
        free(images);

        float model_load_time_ms = (float)(model_load_end - model_load_start) / 1000.0f;
        float run_time_total_ms = (float)(run_total) / 1000.0f;
        float run_time_avg_ms = run_time_total_ms / (float)(repeats * images_len);

        printf("Result:\nTest repeats: %u; Image count: %u; Model load time: %.5f ms\n"
               "Inference time total: %.9f ms; Inference time average: %.9f ms\n",
                repeats, images_len, model_load_time_ms, run_time_total_ms, run_time_avg_ms);

    } else {
        printf("Unrecognised option: '%s'\n", argv[1]);
        fclose(network_file);
        return 1;
    }

    fclose(network_file);
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

uint64_t get_time_us() {
    struct timeval tv = {0};
    gettimeofday(&tv, NULL);
    uint64_t time_us = tv.tv_sec * 1000000L + tv.tv_usec;
    return time_us;
}
