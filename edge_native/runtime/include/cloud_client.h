#ifndef UNISPLIT_CLOUD_CLIENT_H
#define UNISPLIT_CLOUD_CLIENT_H

#include "transport_backend.h"

#include <stddef.h>

typedef struct {
    char status[32];
    int predicted_class;
    char predicted_label[128];
    float timing_total_ms;
} cloud_infer_result_t;

int cloud_client_send_split(
    transport_client_t *transport,
    int split_id,
    const float *activation,
    size_t activation_len,
    const int *shape,
    size_t shape_len,
    int use_quantization,
    const char *model_version,
    cloud_infer_result_t *out,
    char *err,
    size_t err_size
);

int cloud_client_send_split_to_path(
    transport_client_t *transport,
    const char *path,
    int split_id,
    const float *activation,
    size_t activation_len,
    const int *shape,
    size_t shape_len,
    int use_quantization,
    const char *model_version,
    cloud_infer_result_t *out,
    char *err,
    size_t err_size
);

int cloud_client_send_split_k7(
    transport_client_t *transport,
    const float activation[64],
    int use_quantization,
    const char *model_version,
    cloud_infer_result_t *out,
    char *err,
    size_t err_size
);

#endif
