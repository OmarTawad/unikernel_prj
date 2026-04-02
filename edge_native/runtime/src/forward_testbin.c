#include "model_k7.h"
#include "tensor.h"

#include <stdio.h>

int main(int argc, char **argv)
{
    model_k7_params_t model;
    char err[256];
    float input[EDGE_K7_INPUT_LEN];
    float output[EDGE_K7_OUTPUT_LEN];

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <artifacts_dir> <input_f32_bin> <output_f32_bin>\\n", argv[0]);
        return 2;
    }

    if (load_f32_file_exact(argv[2], input, EDGE_K7_INPUT_LEN) != 0) {
        fprintf(stderr, "Failed to load input: %s\\n", argv[2]);
        return 1;
    }

    if (model_k7_load_from_dir(argv[1], &model, err, sizeof(err)) != 0) {
        fprintf(stderr, "Model load failed: %s\\n", err);
        return 1;
    }

    if (model_k7_forward(&model, input, output, err, sizeof(err)) != 0) {
        fprintf(stderr, "Forward failed: %s\\n", err);
        model_k7_free(&model);
        return 1;
    }

    if (write_f32_file(argv[3], output, EDGE_K7_OUTPUT_LEN) != 0) {
        fprintf(stderr, "Failed to write output: %s\\n", argv[3]);
        model_k7_free(&model);
        return 1;
    }

    model_k7_free(&model);
    return 0;
}
