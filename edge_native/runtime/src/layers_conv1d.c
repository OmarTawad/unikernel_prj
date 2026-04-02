#include "layers.h"

void layer_conv1d_valid(
    const float *input,
    int in_channels,
    int in_len,
    const float *weight,
    const float *bias,
    int out_channels,
    int kernel,
    float *output
)
{
    int oc;
    int pos;
    int ic;
    int k;
    int out_len = in_len - kernel + 1;

    for (oc = 0; oc < out_channels; oc++) {
        for (pos = 0; pos < out_len; pos++) {
            float sum = bias ? bias[oc] : 0.0f;
            for (ic = 0; ic < in_channels; ic++) {
                for (k = 0; k < kernel; k++) {
                    int in_idx = ic * in_len + (pos + k);
                    int w_idx = (oc * in_channels + ic) * kernel + k;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
            output[oc * out_len + pos] = sum;
        }
    }
}
