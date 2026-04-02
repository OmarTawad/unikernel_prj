#include "quantize.h"

#include <math.h>

int quantize_int8_symmetric(const float *input, size_t len, int8_t *output, float *scale_out)
{
    size_t i;
    float abs_max = 0.0f;
    float scale;

    if (!input || !output || !scale_out || len == 0) {
        return -1;
    }

    for (i = 0; i < len; i++) {
        float a = fabsf(input[i]);
        if (a > abs_max) {
            abs_max = a;
        }
    }

    if (abs_max < 1e-10f) {
        scale = 1.0f;
    } else {
        scale = abs_max / 127.0f;
    }

    for (i = 0; i < len; i++) {
        float qf = nearbyintf(input[i] / scale);
        int qi = (int) qf;
        if (qi < -127) {
            qi = -127;
        } else if (qi > 127) {
            qi = 127;
        }
        output[i] = (int8_t) qi;
    }

    *scale_out = scale;
    return 0;
}
