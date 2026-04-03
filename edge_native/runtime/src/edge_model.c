#include "edge_model.h"

#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONV1_WEIGHT_COUNT (EDGE_OUT_CH1 * 1 * 3)
#define CONV1_BIAS_COUNT (EDGE_OUT_CH1)
#define BN1_COUNT (EDGE_OUT_CH1)

#define CONV2_WEIGHT_COUNT (EDGE_OUT_CH2 * EDGE_OUT_CH1 * 3)
#define CONV2_BIAS_COUNT (EDGE_OUT_CH2)
#define BN2_COUNT (EDGE_OUT_CH2)

#define FC1_WEIGHT_COUNT (EDGE_FC1_LEN * EDGE_POOL_LEN)
#define FC1_BIAS_COUNT (EDGE_FC1_LEN)

#define FC2_WEIGHT_COUNT (EDGE_LOGITS_LEN * EDGE_FC1_LEN)
#define FC2_BIAS_COUNT (EDGE_LOGITS_LEN)

static void set_error(char *err, size_t err_size, const char *msg)
{
    if (err && err_size > 0) {
        snprintf(err, err_size, "%s", msg);
    }
}

int edge_model_is_supported_split(int split_id)
{
    switch (split_id) {
    case 0:
    case 3:
    case 6:
    case 7:
    case 8:
    case 9:
        return 1;
    default:
        return 0;
    }
}

int edge_model_output_shape_for_split(int split_id, int *shape_out, size_t *ndim_out, size_t *len_out)
{
    if (!shape_out || !ndim_out || !len_out) {
        return -1;
    }

    shape_out[0] = 1;
    shape_out[1] = 1;
    shape_out[2] = 1;

    switch (split_id) {
    case 0:
        *ndim_out = 2;
        shape_out[0] = 1;
        shape_out[1] = EDGE_INPUT_LEN;
        *len_out = EDGE_INPUT_LEN;
        return 0;
    case 3:
        *ndim_out = 2;
        shape_out[0] = EDGE_OUT_CH1;
        shape_out[1] = EDGE_OUT_LEN1;
        *len_out = EDGE_OUT_CH1 * EDGE_OUT_LEN1;
        return 0;
    case 6:
        *ndim_out = 2;
        shape_out[0] = EDGE_OUT_CH2;
        shape_out[1] = EDGE_OUT_LEN2;
        *len_out = EDGE_OUT_CH2 * EDGE_OUT_LEN2;
        return 0;
    case 7:
        *ndim_out = 1;
        shape_out[0] = EDGE_POOL_LEN;
        *len_out = EDGE_POOL_LEN;
        return 0;
    case 8:
        *ndim_out = 1;
        shape_out[0] = EDGE_FC1_LEN;
        *len_out = EDGE_FC1_LEN;
        return 0;
    case 9:
        *ndim_out = 1;
        shape_out[0] = EDGE_LOGITS_LEN;
        *len_out = EDGE_LOGITS_LEN;
        return 0;
    default:
        return -1;
    }
}

static int read_file_text(const char *path, char **out)
{
    FILE *fp = NULL;
    long size;
    char *buf = NULL;
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

static int parse_int_key(const char *manifest_text, const char *key, int *out)
{
    char pattern[64];
    const char *p;

    if (!manifest_text || !key || !out) {
        return -1;
    }

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    p = strstr(manifest_text, pattern);
    if (!p) {
        return -1;
    }

    p = strchr(p, ':');
    if (!p) {
        return -1;
    }
    p++;
    *out = (int) strtol(p, NULL, 10);
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

static int parse_output_shape_from_manifest(const char *manifest_text, edge_model_t *model)
{
    const char *p;
    int values[3];
    size_t n = 0;
    char *endptr = NULL;

    if (!manifest_text || !model) {
        return -1;
    }

    p = strstr(manifest_text, "\"output_shape\"");
    if (!p) {
        return -1;
    }

    p = strchr(p, '[');
    if (!p) {
        return -1;
    }
    p++;

    while (*p && *p != ']' && n < 3) {
        long v;
        while (*p == ' ' || *p == '\t' || *p == ',') {
            p++;
        }
        if (*p == ']') {
            break;
        }
        v = strtol(p, &endptr, 10);
        if (endptr == p) {
            break;
        }
        values[n++] = (int) v;
        p = endptr;
    }

    if (n == 0) {
        return -1;
    }

    model->output_ndim = n;
    model->output_shape[0] = values[0];
    model->output_shape[1] = (n > 1) ? values[1] : 1;
    model->output_shape[2] = (n > 2) ? values[2] : 1;
    model->output_len = (size_t) model->output_shape[0] *
        (size_t) model->output_shape[1] *
        (size_t) model->output_shape[2];
    return 0;
}

static int load_named_tensor(
    const char *artifact_dir,
    const char *base_name,
    size_t expected,
    float **out,
    int required
)
{
    char path[512];
    float *data = NULL;
    size_t count = 0;

    if (!artifact_dir || !base_name || !out) {
        return -1;
    }

    snprintf(path, sizeof(path), "%s/%s.bin", artifact_dir, base_name);
    if (load_f32_file_dynamic(path, &data, &count) != 0) {
        if (required) {
            return -1;
        }
        *out = NULL;
        return 0;
    }

    if (count != expected) {
        free(data);
        return -1;
    }

    *out = data;
    return 0;
}

void edge_model_free(edge_model_t *model)
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

    free(model->fc1_weight);
    free(model->fc1_bias);
    free(model->fc2_weight);
    free(model->fc2_bias);

    memset(model, 0, sizeof(*model));
}

int edge_model_load_from_dir(const char *artifact_dir, edge_model_t *model, char *err, size_t err_size)
{
    char manifest_path[512];
    char *manifest_text = NULL;
    int split_id = -1;
    int need_block1;
    int need_block2;
    int need_fc1;
    int need_fc2;

    if (!artifact_dir || !model) {
        set_error(err, err_size, "Invalid args to edge_model_load_from_dir");
        return -1;
    }

    memset(model, 0, sizeof(*model));
    model->eps = 1e-5f;

    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", artifact_dir);
    if (read_file_text(manifest_path, &manifest_text) != 0) {
        set_error(err, err_size, "Failed to read manifest.json");
        return -1;
    }

    if (parse_int_key(manifest_text, "split_id", &split_id) != 0 || !edge_model_is_supported_split(split_id)) {
        free(manifest_text);
        set_error(err, err_size, "Manifest split_id is missing or unsupported");
        return -1;
    }

    model->split_id = split_id;
    model->eps = parse_eps_from_manifest(manifest_text);
    if (edge_model_output_shape_for_split(split_id, model->output_shape, &model->output_ndim, &model->output_len) != 0) {
        free(manifest_text);
        set_error(err, err_size, "Failed to map split output shape");
        return -1;
    }
    (void) parse_output_shape_from_manifest(manifest_text, model);
    free(manifest_text);

    need_block1 = (split_id >= 3);
    need_block2 = (split_id >= 6);
    need_fc1 = (split_id >= 8);
    need_fc2 = (split_id >= 9);

    if (load_named_tensor(artifact_dir, "conv1_weight", CONV1_WEIGHT_COUNT, &model->conv1_weight, need_block1) != 0 ||
        load_named_tensor(artifact_dir, "conv1_bias", CONV1_BIAS_COUNT, &model->conv1_bias, need_block1) != 0 ||
        load_named_tensor(artifact_dir, "bn1_gamma", BN1_COUNT, &model->bn1_gamma, need_block1) != 0 ||
        load_named_tensor(artifact_dir, "bn1_beta", BN1_COUNT, &model->bn1_beta, need_block1) != 0 ||
        load_named_tensor(artifact_dir, "bn1_running_mean", BN1_COUNT, &model->bn1_mean, need_block1) != 0 ||
        load_named_tensor(artifact_dir, "bn1_running_var", BN1_COUNT, &model->bn1_var, need_block1) != 0 ||
        load_named_tensor(artifact_dir, "conv2_weight", CONV2_WEIGHT_COUNT, &model->conv2_weight, need_block2) != 0 ||
        load_named_tensor(artifact_dir, "conv2_bias", CONV2_BIAS_COUNT, &model->conv2_bias, need_block2) != 0 ||
        load_named_tensor(artifact_dir, "bn2_gamma", BN2_COUNT, &model->bn2_gamma, need_block2) != 0 ||
        load_named_tensor(artifact_dir, "bn2_beta", BN2_COUNT, &model->bn2_beta, need_block2) != 0 ||
        load_named_tensor(artifact_dir, "bn2_running_mean", BN2_COUNT, &model->bn2_mean, need_block2) != 0 ||
        load_named_tensor(artifact_dir, "bn2_running_var", BN2_COUNT, &model->bn2_var, need_block2) != 0 ||
        load_named_tensor(artifact_dir, "fc1_weight", FC1_WEIGHT_COUNT, &model->fc1_weight, need_fc1) != 0 ||
        load_named_tensor(artifact_dir, "fc1_bias", FC1_BIAS_COUNT, &model->fc1_bias, need_fc1) != 0 ||
        load_named_tensor(artifact_dir, "fc2_weight", FC2_WEIGHT_COUNT, &model->fc2_weight, need_fc2) != 0 ||
        load_named_tensor(artifact_dir, "fc2_bias", FC2_BIAS_COUNT, &model->fc2_bias, need_fc2) != 0) {
        edge_model_free(model);
        set_error(err, err_size, "Failed to load one or more tensors for split");
        return -1;
    }

    return 0;
}
