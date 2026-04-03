#include "layers.h"

void layer_linear(
    const float *input,
    int in_features,
    const float *weight,
    const float *bias,
    int out_features,
    float *output
)
{
    int o;
    int i;

    for (o = 0; o < out_features; o++) {
        float sum = bias ? bias[o] : 0.0f;
        for (i = 0; i < in_features; i++) {
            sum += input[i] * weight[o * in_features + i];
        }
        output[o] = sum;
    }
}
