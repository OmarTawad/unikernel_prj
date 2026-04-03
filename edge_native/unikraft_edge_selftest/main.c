#include "controller.h"
#include "edge_model.h"
#include "edge_runtime.h"
#include "transport_backend.h"

#include <stdio.h>
#include <string.h>

#define EDGE_OUTPUT_MAX (EDGE_OUT_CH2 * EDGE_OUT_LEN2)

static float g_conv1_weight[EDGE_OUT_CH1 * 1 * 3];
static float g_conv1_bias[EDGE_OUT_CH1];
static float g_bn1_gamma[EDGE_OUT_CH1];
static float g_bn1_beta[EDGE_OUT_CH1];
static float g_bn1_mean[EDGE_OUT_CH1];
static float g_bn1_var[EDGE_OUT_CH1];

static float g_conv2_weight[EDGE_OUT_CH2 * EDGE_OUT_CH1 * 3];
static float g_conv2_bias[EDGE_OUT_CH2];
static float g_bn2_gamma[EDGE_OUT_CH2];
static float g_bn2_beta[EDGE_OUT_CH2];
static float g_bn2_mean[EDGE_OUT_CH2];
static float g_bn2_var[EDGE_OUT_CH2];

static float g_fc1_weight[EDGE_FC1_LEN * EDGE_POOL_LEN];
static float g_fc1_bias[EDGE_FC1_LEN];
static float g_input[EDGE_INPUT_LEN];
static float g_split_output[EDGE_OUTPUT_MAX];

static void init_deterministic_tensors(void)
{
    int i;
    memset(g_conv1_weight, 0, sizeof(g_conv1_weight));
    memset(g_conv2_weight, 0, sizeof(g_conv2_weight));
    memset(g_fc1_weight, 0, sizeof(g_fc1_weight));

    for (i = 0; i < EDGE_OUT_CH1; i++) {
        g_conv1_bias[i] = 1.0f;
        g_bn1_gamma[i] = 1.0f;
        g_bn1_beta[i] = 0.0f;
        g_bn1_mean[i] = 0.0f;
        g_bn1_var[i] = 1.0f;
    }
    for (i = 0; i < EDGE_OUT_CH2; i++) {
        g_conv2_bias[i] = 2.0f;
        g_bn2_gamma[i] = 1.0f;
        g_bn2_beta[i] = 0.0f;
        g_bn2_mean[i] = 0.0f;
        g_bn2_var[i] = 1.0f;
    }
    for (i = 0; i < EDGE_FC1_LEN; i++) {
        g_fc1_bias[i] = 3.0f;
    }
}

static void setup_model_for_split(edge_model_t *model, int split_id)
{
    memset(model, 0, sizeof(*model));
    model->split_id = split_id;
    model->eps = 1e-5f;
    (void) edge_model_output_shape_for_split(split_id, model->output_shape, &model->output_ndim, &model->output_len);

    model->conv1_weight = g_conv1_weight;
    model->conv1_bias = g_conv1_bias;
    model->bn1_gamma = g_bn1_gamma;
    model->bn1_beta = g_bn1_beta;
    model->bn1_mean = g_bn1_mean;
    model->bn1_var = g_bn1_var;

    model->conv2_weight = g_conv2_weight;
    model->conv2_bias = g_conv2_bias;
    model->bn2_gamma = g_bn2_gamma;
    model->bn2_beta = g_bn2_beta;
    model->bn2_mean = g_bn2_mean;
    model->bn2_var = g_bn2_var;

    model->fc1_weight = g_fc1_weight;
    model->fc1_bias = g_fc1_bias;
}

static int run_split_selftest(const float *input, int split_id, float *out_buf, size_t *out_len)
{
    edge_model_t model;
    char err[128];

    setup_model_for_split(&model, split_id);
    if (edge_runtime_forward(&model, input, out_buf, EDGE_OUTPUT_MAX, out_len, err, sizeof(err)) != 0) {
        printf("UK_SELFTEST_FAIL split=%d err=%s\n", split_id, err);
        return -1;
    }
    printf("UK_SELFTEST_SPLIT_OK split=%d len=%zu\n", split_id, *out_len);
    return 0;
}

int main(int argc, char *argv[])
{
    size_t out_len = 0;
    int i;

    feasible_split_set_t feasible;
    int selected = -1;

    transport_client_t t;
    char *resp = NULL;
    char err[128];
    int splits[3] = {3, 7, 8};

    (void) argc;
    (void) argv;

    init_deterministic_tensors();
    for (i = 0; i < EDGE_INPUT_LEN; i++) {
        g_input[i] = (float) i / (float) EDGE_INPUT_LEN;
    }

    if (run_split_selftest(g_input, 3, g_split_output, &out_len) != 0) {
        return 1;
    }
    if (run_split_selftest(g_input, 7, g_split_output, &out_len) != 0) {
        return 1;
    }
    if (run_split_selftest(g_input, 8, g_split_output, &out_len) != 0) {
        return 1;
    }
    printf("UK_SELFTEST_EDGE_OK\n");

    if (feasible_split_set_init(&feasible, splits, 3) != 0) {
        printf("UK_SELFTEST_FAIL controller_init\n");
        return 1;
    }
    selected = feasible.split_ids[0];
    if (!feasible_split_set_contains(&feasible, selected)) {
        printf("UK_SELFTEST_FAIL controller_contains\n");
        return 1;
    }
    printf("UK_SELFTEST_CTRL_OK selected=%d count=%zu\n", selected, feasible.count);

    if (transport_create_by_name("ukstub", "ukstub://ok", 1, &t, err, sizeof(err)) != 0) {
        printf("UK_SELFTEST_FAIL transport_create err=%s\n", err);
        return 1;
    }
    printf("UK_SELFTEST_TRANSPORT=ukstub\n");

    if (transport_client_post_json(&t, "/infer/split", "{\"request_id\":\"uk-selftest\"}", &resp) != 0) {
        transport_client_destroy(&t);
        printf("UK_SELFTEST_FAIL transport_post\n");
        return 1;
    }
    if (!resp || strstr(resp, "\"status\":\"ok\"") == NULL) {
        transport_response_free(resp);
        transport_client_destroy(&t);
        printf("UK_SELFTEST_FAIL transport_resp\n");
        return 1;
    }
    transport_response_free(resp);
    transport_client_destroy(&t);
    printf("UK_SELFTEST_TRANSPORT_OK\n");

    printf("UK_SELFTEST_DONE\n");
    return 0;
}
