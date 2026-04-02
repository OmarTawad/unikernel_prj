#ifndef UNISPLIT_MODEL_K7_H
#define UNISPLIT_MODEL_K7_H

#include <stddef.h>

#define EDGE_K7_INPUT_LEN 80
#define EDGE_K7_OUT1_CH 32
#define EDGE_K7_OUT1_LEN 78
#define EDGE_K7_OUT2_CH 64
#define EDGE_K7_OUT2_LEN 76
#define EDGE_K7_OUTPUT_LEN 64

typedef struct {
    float eps;

    float *conv1_weight; /* [32,1,3] */
    float *conv1_bias;   /* [32] */
    float *bn1_gamma;    /* [32] */
    float *bn1_beta;     /* [32] */
    float *bn1_mean;     /* [32] */
    float *bn1_var;      /* [32] */

    float *conv2_weight; /* [64,32,3] */
    float *conv2_bias;   /* [64] */
    float *bn2_gamma;    /* [64] */
    float *bn2_beta;     /* [64] */
    float *bn2_mean;     /* [64] */
    float *bn2_var;      /* [64] */
} model_k7_params_t;

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
