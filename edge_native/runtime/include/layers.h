#ifndef UNISPLIT_LAYERS_H
#define UNISPLIT_LAYERS_H

void layer_conv1d_valid(
    const float *input,
    int in_channels,
    int in_len,
    const float *weight,
    const float *bias,
    int out_channels,
    int kernel,
    float *output
);

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
);

void layer_relu(const float *input, int len, float *output);
void layer_global_avgpool(const float *input, int channels, int len, float *output);

#endif
