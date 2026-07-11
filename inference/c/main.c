#include <stdio.h>

#include "network.h"

int main(int argc, char * argv[]) {
    assert(argc > 1);

    const char * network_name = argv[1];

    FILE * network_file = fopen(network_name, "r");


    network_t net = {0};
    network_init(&net, 3);
    network_load_layers(&net, network_file);

    puts("loading complete!");

    network_free(&net);

    return 0;
}
