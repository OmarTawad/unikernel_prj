#ifndef UNISPLIT_EDGE_RUNTIME_H
#define UNISPLIT_EDGE_RUNTIME_H

#include "edge_model.h"

#include <stddef.h>

int edge_runtime_forward(
    const edge_model_t *model,
    const float input[EDGE_INPUT_LEN],
    float *output,
    size_t output_cap,
    size_t *output_len,
    char *err,
    size_t err_size
);

#endif
