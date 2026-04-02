#include "layers.h"

void layer_global_avgpool(const float *input, int channels, int len, float *output)
{
    int c;
    int i;

    for (c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (i = 0; i < len; i++) {
            sum += input[c * len + i];
        }
        output[c] = sum / (float) len;
    }
}
