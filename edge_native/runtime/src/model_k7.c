#include "model_k7.h"

#include "layers.h"
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONV1_WEIGHT_COUNT (EDGE_K7_OUT1_CH * 1 * 3)
#define CONV1_BIAS_COUNT (EDGE_K7_OUT1_CH)
#define BN1_COUNT (EDGE_K7_OUT1_CH)

#define CONV2_WEIGHT_COUNT (EDGE_K7_OUT2_CH * EDGE_K7_OUT1_CH * 3)
#define CONV2_BIAS_COUNT (EDGE_K7_OUT2_CH)
#define BN2_COUNT (EDGE_K7_OUT2_CH)

static void set_error(char *err, size_t err_size, const char *msg)
{
    if (err && err_size > 0) {
        snprintf(err, err_size, "%s", msg);
    }
}

static int read_file_text(const char *path, char **out)
{
    FILE *fp;
    long size;
    char *buf;
    size_t read_bytes;

    if (!path || !out) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }

    size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return -1;
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }

    buf = (char *) malloc((size_t) size + 1);
    if (!buf) {
        fclose(fp);
        return -1;
    }

    read_bytes = fread(buf, 1, (size_t) size, fp);
    fclose(fp);

    if (read_bytes != (size_t) size) {
        free(buf);
        return -1;
    }

    buf[size] = '\0';
    *out = buf;
    return 0;
}

static float parse_eps_from_manifest(const char *manifest_text)
{
    const char *p;
    float eps = 1e-5f;

    if (!manifest_text) {
        return eps;
    }

    p = strstr(manifest_text, "\"eps\"");
    if (!p) {
        return eps;
    }

    p = strchr(p, ':');
    if (!p) {
        return eps;
    }

    p++;
    eps = strtof(p, NULL);
    if (eps <= 0.0f) {
        eps = 1e-5f;
    }

    return eps;
}

static int load_named_tensor(const char *artifact_dir, const char *base_name, size_t expected, float **out)
{
    char path[512];
    size_t count = 0;
    float *data = NULL;

    snprintf(path, sizeof(path), "%s/%s.bin", artifact_dir, base_name);
    if (load_f32_file_dynamic(path, &data, &count) != 0) {
        return -1;
    }

    if (count != expected) {
        free(data);
        return -1;
    }

    *out = data;
    return 0;
}

int model_k7_load_from_dir(const char *artifact_dir, model_k7_params_t *model, char *err, size_t err_size)
{
    char manifest_path[512];
    char *manifest_text = NULL;

    if (!artifact_dir || !model) {
        set_error(err, err_size, "Invalid arguments to model_k7_load_from_dir");
        return -1;
    }

    memset(model, 0, sizeof(*model));

    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", artifact_dir);
    if (read_file_text(manifest_path, &manifest_text) != 0) {
        set_error(err, err_size, "Failed to read manifest.json");
        return -1;
    }

    model->eps = parse_eps_from_manifest(manifest_text);
    free(manifest_text);

    if (load_named_tensor(artifact_dir, "conv1_weight", CONV1_WEIGHT_COUNT, &model->conv1_weight) != 0 ||
        load_named_tensor(artifact_dir, "conv1_bias", CONV1_BIAS_COUNT, &model->conv1_bias) != 0 ||
        load_named_tensor(artifact_dir, "bn1_gamma", BN1_COUNT, &model->bn1_gamma) != 0 ||
        load_named_tensor(artifact_dir, "bn1_beta", BN1_COUNT, &model->bn1_beta) != 0 ||
        load_named_tensor(artifact_dir, "bn1_running_mean", BN1_COUNT, &model->bn1_mean) != 0 ||
        load_named_tensor(artifact_dir, "bn1_running_var", BN1_COUNT, &model->bn1_var) != 0 ||
        load_named_tensor(artifact_dir, "conv2_weight", CONV2_WEIGHT_COUNT, &model->conv2_weight) != 0 ||
        load_named_tensor(artifact_dir, "conv2_bias", CONV2_BIAS_COUNT, &model->conv2_bias) != 0 ||
        load_named_tensor(artifact_dir, "bn2_gamma", BN2_COUNT, &model->bn2_gamma) != 0 ||
        load_named_tensor(artifact_dir, "bn2_beta", BN2_COUNT, &model->bn2_beta) != 0 ||
        load_named_tensor(artifact_dir, "bn2_running_mean", BN2_COUNT, &model->bn2_mean) != 0 ||
        load_named_tensor(artifact_dir, "bn2_running_var", BN2_COUNT, &model->bn2_var) != 0) {
        model_k7_free(model);
        set_error(err, err_size, "Failed to load one or more edge_k7 tensors");
        return -1;
    }

    return 0;
}

void model_k7_free(model_k7_params_t *model)
{
    if (!model) {
        return;
    }

    free(model->conv1_weight);
    free(model->conv1_bias);
    free(model->bn1_gamma);
    free(model->bn1_beta);
    free(model->bn1_mean);
    free(model->bn1_var);
    free(model->conv2_weight);
    free(model->conv2_bias);
    free(model->bn2_gamma);
    free(model->bn2_beta);
    free(model->bn2_mean);
    free(model->bn2_var);

    memset(model, 0, sizeof(*model));
}

int model_k7_forward(
    const model_k7_params_t *model,
    const float input[EDGE_K7_INPUT_LEN],
    float output[EDGE_K7_OUTPUT_LEN],
    char *err,
    size_t err_size
)
{
    float conv1_out[EDGE_K7_OUT1_CH * EDGE_K7_OUT1_LEN];
    float bn1_out[EDGE_K7_OUT1_CH * EDGE_K7_OUT1_LEN];
    float relu1_out[EDGE_K7_OUT1_CH * EDGE_K7_OUT1_LEN];

    float conv2_out[EDGE_K7_OUT2_CH * EDGE_K7_OUT2_LEN];
    float bn2_out[EDGE_K7_OUT2_CH * EDGE_K7_OUT2_LEN];
    float relu2_out[EDGE_K7_OUT2_CH * EDGE_K7_OUT2_LEN];

    if (!model || !input || !output) {
        set_error(err, err_size, "Invalid arguments to model_k7_forward");
        return -1;
    }

    if (!model->conv1_weight || !model->conv2_weight) {
        set_error(err, err_size, "Model parameters not loaded");
        return -1;
    }

    layer_conv1d_valid(
        input,
        1,
        EDGE_K7_INPUT_LEN,
        model->conv1_weight,
        model->conv1_bias,
        EDGE_K7_OUT1_CH,
        3,
        conv1_out
    );

    layer_batchnorm_eval(
        conv1_out,
        EDGE_K7_OUT1_CH,
        EDGE_K7_OUT1_LEN,
        model->bn1_gamma,
        model->bn1_beta,
        model->bn1_mean,
        model->bn1_var,
        model->eps,
        bn1_out
    );

    layer_relu(bn1_out, EDGE_K7_OUT1_CH * EDGE_K7_OUT1_LEN, relu1_out);

    layer_conv1d_valid(
        relu1_out,
        EDGE_K7_OUT1_CH,
        EDGE_K7_OUT1_LEN,
        model->conv2_weight,
        model->conv2_bias,
        EDGE_K7_OUT2_CH,
        3,
        conv2_out
    );

    layer_batchnorm_eval(
        conv2_out,
        EDGE_K7_OUT2_CH,
        EDGE_K7_OUT2_LEN,
        model->bn2_gamma,
        model->bn2_beta,
        model->bn2_mean,
        model->bn2_var,
        model->eps,
        bn2_out
    );

    layer_relu(bn2_out, EDGE_K7_OUT2_CH * EDGE_K7_OUT2_LEN, relu2_out);

    layer_global_avgpool(relu2_out, EDGE_K7_OUT2_CH, EDGE_K7_OUT2_LEN, output);
    return 0;
}
