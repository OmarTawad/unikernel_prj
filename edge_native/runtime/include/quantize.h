#ifndef UNISPLIT_QUANTIZE_H
#define UNISPLIT_QUANTIZE_H

#include <stddef.h>
#include <stdint.h>

int quantize_int8_symmetric(const float *input, size_t len, int8_t *output, float *scale_out);

#endif
