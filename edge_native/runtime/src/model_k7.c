#include "model_k7.h"

#include "edge_runtime.h"

#include <stdio.h>

int model_k7_load_from_dir(const char *artifact_dir, model_k7_params_t *model, char *err, size_t err_size)
{
    if (!model) {
        return -1;
    }
    return edge_model_load_from_dir(artifact_dir, &model->inner, err, err_size);
}

void model_k7_free(model_k7_params_t *model)
{
    if (!model) {
        return;
    }
    edge_model_free(&model->inner);
}

int model_k7_forward(
    const model_k7_params_t *model,
    const float input[EDGE_K7_INPUT_LEN],
    float output[EDGE_K7_OUTPUT_LEN],
    char *err,
    size_t err_size
)
{
    size_t out_len = 0;
    if (!model) {
        return -1;
    }
    if (model->inner.split_id != 7) {
        if (err && err_size > 0) {
            snprintf(err, err_size, "model_k7_forward requires split_id=7 artifact");
        }
        return -1;
    }
    return edge_runtime_forward(
        &model->inner,
        input,
        output,
        EDGE_K7_OUTPUT_LEN,
        &out_len,
        err,
        err_size
    ) == 0 && out_len == EDGE_K7_OUTPUT_LEN ? 0 : -1;
}
