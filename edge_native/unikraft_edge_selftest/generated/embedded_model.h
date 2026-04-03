#ifndef UNISPLIT_EMBEDDED_MODEL_H
#define UNISPLIT_EMBEDDED_MODEL_H

#include "edge_model.h"

#include <stddef.h>

int edge_model_load_embedded(edge_model_t *model, int split_id, char *err, size_t err_size);
const float *edge_embedded_reference_input(void);
size_t edge_embedded_reference_input_len(void);
const char *edge_embedded_artifact_strategy(void);

#endif
