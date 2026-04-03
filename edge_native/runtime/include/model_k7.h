#ifndef UNISPLIT_MODEL_K7_H
#define UNISPLIT_MODEL_K7_H

#include "edge_model.h"

#include <stddef.h>

typedef struct {
    edge_model_t inner;
} model_k7_params_t;

#define EDGE_K7_INPUT_LEN EDGE_INPUT_LEN
#define EDGE_K7_OUTPUT_LEN EDGE_POOL_LEN

int model_k7_load_from_dir(const char *artifact_dir, model_k7_params_t *model, char *err, size_t err_size);
void model_k7_free(model_k7_params_t *model);

int model_k7_forward(
    const model_k7_params_t *model,
    const float input[EDGE_K7_INPUT_LEN],
    float output[EDGE_K7_OUTPUT_LEN],
    char *err,
    size_t err_size
);

#endif
