#include "layers.h"

#include <math.h>

void layer_batchnorm_eval(
    const float *input,
    int channels,
    int len,
    const float *gamma,
    const float *beta,
    const float *mean,
    const float *var,
    float eps,
    float *output
)
{
    int c;
    int i;

    for (c = 0; c < channels; c++) {
        float inv_std = 1.0f / sqrtf(var[c] + eps);
        float g = gamma[c];
        float b = beta[c];
        float m = mean[c];
        for (i = 0; i < len; i++) {
            int idx = c * len + i;
            float x = input[idx];
            output[idx] = ((x - m) * inv_std) * g + b;
        }
    }
}
