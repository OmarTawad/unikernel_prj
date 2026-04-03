#ifndef UNISPLIT_EDGE_MODEL_H
#define UNISPLIT_EDGE_MODEL_H

#include <stddef.h>

#define EDGE_INPUT_LEN 80
#define EDGE_SPLIT_COUNT 6

#define EDGE_OUT_CH1 32
#define EDGE_OUT_LEN1 78

#define EDGE_OUT_CH2 64
#define EDGE_OUT_LEN2 76

#define EDGE_POOL_LEN 64
#define EDGE_FC1_LEN 128
#define EDGE_LOGITS_LEN 34

typedef struct {
    int split_id;
    float eps;
    int output_shape[3];
    size_t output_ndim;
    size_t output_len;

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

    float *fc1_weight; /* [128,64] */
    float *fc1_bias;   /* [128] */
    float *fc2_weight; /* [34,128] */
    float *fc2_bias;   /* [34] */
} edge_model_t;

int edge_model_is_supported_split(int split_id);
int edge_model_output_shape_for_split(int split_id, int *shape_out, size_t *ndim_out, size_t *len_out);

int edge_model_load_from_dir(const char *artifact_dir, edge_model_t *model, char *err, size_t err_size);
void edge_model_free(edge_model_t *model);

#endif
