#include "layers.h"

void layer_relu(const float *input, int len, float *output)
{
    int i;
    for (i = 0; i < len; i++) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : 0.0f;
    }
}
