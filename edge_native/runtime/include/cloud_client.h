#ifndef UNISPLIT_CLOUD_CLIENT_H
#define UNISPLIT_CLOUD_CLIENT_H

#include "transport.h"

typedef struct {
    char status[32];
    int predicted_class;
    char predicted_label[128];
    float timing_total_ms;
} cloud_infer_result_t;

int cloud_client_send_split_k7(
    const transport_cfg_t *transport,
    const float activation[64],
    int use_quantization,
    const char *model_version,
    cloud_infer_result_t *out,
    char *err,
    unsigned long err_size
);

#endif
