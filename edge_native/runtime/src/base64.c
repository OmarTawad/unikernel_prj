#include "base64.h"

#include <stdlib.h>

static const char BASE64_TABLE[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int base64_encode(const uint8_t *input, size_t len, char **out_b64, size_t *out_len)
{
    size_t i;
    size_t j = 0;
    size_t enc_len;
    char *out;

    if (!input || !out_b64 || !out_len) {
        return -1;
    }

    enc_len = 4 * ((len + 2) / 3);
    out = (char *) malloc(enc_len + 1);
    if (!out) {
        return -1;
    }

    for (i = 0; i < len; i += 3) {
        uint32_t octet_a = input[i];
        uint32_t octet_b = (i + 1 < len) ? input[i + 1] : 0;
        uint32_t octet_c = (i + 2 < len) ? input[i + 2] : 0;
        uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;

        out[j++] = BASE64_TABLE[(triple >> 18) & 0x3F];
        out[j++] = BASE64_TABLE[(triple >> 12) & 0x3F];
        out[j++] = (i + 1 < len) ? BASE64_TABLE[(triple >> 6) & 0x3F] : '=';
        out[j++] = (i + 2 < len) ? BASE64_TABLE[triple & 0x3F] : '=';
    }

    out[j] = '\0';
    *out_b64 = out;
    *out_len = j;
    return 0;
}
