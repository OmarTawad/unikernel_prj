#ifndef UNISPLIT_BASE64_H
#define UNISPLIT_BASE64_H

#include <stddef.h>
#include <stdint.h>

int base64_encode(const uint8_t *input, size_t len, char **out_b64, size_t *out_len);

#endif
