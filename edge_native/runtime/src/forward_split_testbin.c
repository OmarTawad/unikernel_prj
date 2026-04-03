#include "edge_model.h"
#include "edge_runtime.h"
#include "tensor.h"

#include <stdio.h>

#define EDGE_OUTPUT_MAX (EDGE_OUT_CH2 * EDGE_OUT_LEN2)

int main(int argc, char **argv)
{
    edge_model_t model;
    char err[256];
    float input[EDGE_INPUT_LEN];
    float output[EDGE_OUTPUT_MAX];
    size_t output_len = 0;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <artifacts_dir> <input_f32_bin> <output_f32_bin>\\n", argv[0]);
        return 2;
    }

    if (load_f32_file_exact(argv[2], input, EDGE_INPUT_LEN) != 0) {
        fprintf(stderr, "Failed to load input: %s\\n", argv[2]);
        return 1;
    }
    if (edge_model_load_from_dir(argv[1], &model, err, sizeof(err)) != 0) {
        fprintf(stderr, "Model load failed: %s\\n", err);
        return 1;
    }
    if (edge_runtime_forward(&model, input, output, EDGE_OUTPUT_MAX, &output_len, err, sizeof(err)) != 0) {
        fprintf(stderr, "Forward failed: %s\\n", err);
        edge_model_free(&model);
        return 1;
    }
    if (write_f32_file(argv[3], output, output_len) != 0) {
        fprintf(stderr, "Failed to write output: %s\\n", argv[3]);
        edge_model_free(&model);
        return 1;
    }

    edge_model_free(&model);
    return 0;
}
