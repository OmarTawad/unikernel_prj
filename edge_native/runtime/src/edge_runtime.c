#include "edge_runtime.h"

#include "layers.h"

#include <stdio.h>
#include <string.h>

static void set_error(char *err, size_t err_size, const char *msg)
{
    if (err && err_size > 0) {
        snprintf(err, err_size, "%s", msg);
    }
}

static int check_output_cap(size_t output_cap, size_t need)
{
    return output_cap >= need ? 0 : -1;
}

int edge_runtime_forward(
    const edge_model_t *model,
    const float input[EDGE_INPUT_LEN],
    float *output,
    size_t output_cap,
    size_t *output_len,
    char *err,
    size_t err_size
)
{
    float conv1_out[EDGE_OUT_CH1 * EDGE_OUT_LEN1];
    float bn1_out[EDGE_OUT_CH1 * EDGE_OUT_LEN1];
    float relu1_out[EDGE_OUT_CH1 * EDGE_OUT_LEN1];

    float conv2_out[EDGE_OUT_CH2 * EDGE_OUT_LEN2];
    float bn2_out[EDGE_OUT_CH2 * EDGE_OUT_LEN2];
    float relu2_out[EDGE_OUT_CH2 * EDGE_OUT_LEN2];

    float pool_out[EDGE_POOL_LEN];
    float fc1_out[EDGE_FC1_LEN];
    float logits[EDGE_LOGITS_LEN];

    if (!model || !input || !output || !output_len) {
        set_error(err, err_size, "Invalid args to edge_runtime_forward");
        return -1;
    }

    switch (model->split_id) {
    case 0:
        if (check_output_cap(output_cap, EDGE_INPUT_LEN) != 0) {
            set_error(err, err_size, "Output buffer too small for split=0");
            return -1;
        }
        memcpy(output, input, sizeof(float) * EDGE_INPUT_LEN);
        *output_len = EDGE_INPUT_LEN;
        return 0;

    case 3:
    case 6:
    case 7:
    case 8:
    case 9:
        if (!model->conv1_weight || !model->bn1_gamma) {
            set_error(err, err_size, "Split requires block1 tensors but they are missing");
            return -1;
        }
        break;
    default:
        set_error(err, err_size, "Unsupported split_id in edge_runtime_forward");
        return -1;
    }

    layer_conv1d_valid(
        input, 1, EDGE_INPUT_LEN,
        model->conv1_weight, model->conv1_bias,
        EDGE_OUT_CH1, 3,
        conv1_out
    );

    layer_batchnorm_eval(
        conv1_out,
        EDGE_OUT_CH1,
        EDGE_OUT_LEN1,
        model->bn1_gamma,
        model->bn1_beta,
        model->bn1_mean,
        model->bn1_var,
        model->eps,
        bn1_out
    );
    layer_relu(bn1_out, EDGE_OUT_CH1 * EDGE_OUT_LEN1, relu1_out);

    if (model->split_id == 3) {
        if (check_output_cap(output_cap, EDGE_OUT_CH1 * EDGE_OUT_LEN1) != 0) {
            set_error(err, err_size, "Output buffer too small for split=3");
            return -1;
        }
        memcpy(output, relu1_out, sizeof(float) * EDGE_OUT_CH1 * EDGE_OUT_LEN1);
        *output_len = EDGE_OUT_CH1 * EDGE_OUT_LEN1;
        return 0;
    }

    if (!model->conv2_weight || !model->bn2_gamma) {
        set_error(err, err_size, "Split requires block2 tensors but they are missing");
        return -1;
    }

    layer_conv1d_valid(
        relu1_out,
        EDGE_OUT_CH1,
        EDGE_OUT_LEN1,
        model->conv2_weight,
        model->conv2_bias,
        EDGE_OUT_CH2,
        3,
        conv2_out
    );
    layer_batchnorm_eval(
        conv2_out,
        EDGE_OUT_CH2,
        EDGE_OUT_LEN2,
        model->bn2_gamma,
        model->bn2_beta,
        model->bn2_mean,
        model->bn2_var,
        model->eps,
        bn2_out
    );
    layer_relu(bn2_out, EDGE_OUT_CH2 * EDGE_OUT_LEN2, relu2_out);

    if (model->split_id == 6) {
        if (check_output_cap(output_cap, EDGE_OUT_CH2 * EDGE_OUT_LEN2) != 0) {
            set_error(err, err_size, "Output buffer too small for split=6");
            return -1;
        }
        memcpy(output, relu2_out, sizeof(float) * EDGE_OUT_CH2 * EDGE_OUT_LEN2);
        *output_len = EDGE_OUT_CH2 * EDGE_OUT_LEN2;
        return 0;
    }

    layer_global_avgpool(relu2_out, EDGE_OUT_CH2, EDGE_OUT_LEN2, pool_out);
    if (model->split_id == 7) {
        if (check_output_cap(output_cap, EDGE_POOL_LEN) != 0) {
            set_error(err, err_size, "Output buffer too small for split=7");
            return -1;
        }
        memcpy(output, pool_out, sizeof(float) * EDGE_POOL_LEN);
        *output_len = EDGE_POOL_LEN;
        return 0;
    }

    if (!model->fc1_weight || !model->fc1_bias) {
        set_error(err, err_size, "Split requires fc1 tensors but they are missing");
        return -1;
    }
    layer_linear(pool_out, EDGE_POOL_LEN, model->fc1_weight, model->fc1_bias, EDGE_FC1_LEN, fc1_out);
    layer_relu(fc1_out, EDGE_FC1_LEN, fc1_out);
    if (model->split_id == 8) {
        if (check_output_cap(output_cap, EDGE_FC1_LEN) != 0) {
            set_error(err, err_size, "Output buffer too small for split=8");
            return -1;
        }
        memcpy(output, fc1_out, sizeof(float) * EDGE_FC1_LEN);
        *output_len = EDGE_FC1_LEN;
        return 0;
    }

    if (!model->fc2_weight || !model->fc2_bias) {
        set_error(err, err_size, "Split requires fc2 tensors but they are missing");
        return -1;
    }
    layer_linear(fc1_out, EDGE_FC1_LEN, model->fc2_weight, model->fc2_bias, EDGE_LOGITS_LEN, logits);

    if (check_output_cap(output_cap, EDGE_LOGITS_LEN) != 0) {
        set_error(err, err_size, "Output buffer too small for split=9");
        return -1;
    }
    memcpy(output, logits, sizeof(float) * EDGE_LOGITS_LEN);
    *output_len = EDGE_LOGITS_LEN;
    return 0;
}
