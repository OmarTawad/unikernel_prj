#include "cloud_client.h"

#include "base64.h"
#include "quantize.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void set_error(char *err, size_t err_size, const char *msg)
{
    if (err && err_size > 0) {
        snprintf(err, err_size, "%s", msg);
    }
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1000000.0;
}

static void make_request_id(char *dst, size_t size)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    snprintf(dst, size, "edge-split-%ld-%ld", (long) ts.tv_sec, (long) ts.tv_nsec);
}

static int json_extract_string(const char *json, const char *key, char *out, size_t out_size)
{
    char pattern[64];
    const char *p;
    const char *q;
    size_t n;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    p = strstr(json, pattern);
    if (!p) {
        return -1;
    }
    p = strchr(p, ':');
    if (!p) {
        return -1;
    }
    p++;
    while (*p == ' ' || *p == '\t') {
        p++;
    }
    if (*p != '"') {
        return -1;
    }
    p++;
    q = strchr(p, '"');
    if (!q) {
        return -1;
    }
    n = (size_t) (q - p);
    if (n >= out_size) {
        n = out_size - 1;
    }
    memcpy(out, p, n);
    out[n] = '\0';
    return 0;
}

static int json_extract_int(const char *json, const char *key, int *out)
{
    char pattern[64];
    const char *p;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    p = strstr(json, pattern);
    if (!p) {
        return -1;
    }
    p = strchr(p, ':');
    if (!p) {
        return -1;
    }
    *out = (int) strtol(p + 1, NULL, 10);
    return 0;
}

static int json_extract_float(const char *json, const char *key, float *out)
{
    char pattern[64];
    const char *p;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    p = strstr(json, pattern);
    if (!p) {
        return -1;
    }
    p = strchr(p, ':');
    if (!p) {
        return -1;
    }
    *out = strtof(p + 1, NULL);
    return 0;
}

static int append_shape_json(char *dst, size_t cap, const int *shape, size_t shape_len)
{
    size_t pos = 0;
    size_t i;
    int written;

    if (!dst || !shape || shape_len == 0 || cap < 3) {
        return -1;
    }
    dst[pos++] = '[';
    for (i = 0; i < shape_len; i++) {
        if (i > 0) {
            if (pos + 1 >= cap) {
                return -1;
            }
            dst[pos++] = ',';
        }
        written = snprintf(dst + pos, cap - pos, "%d", shape[i]);
        if (written < 0 || (size_t) written >= cap - pos) {
            return -1;
        }
        pos += (size_t) written;
    }
    if (pos + 2 > cap) {
        return -1;
    }
    dst[pos++] = ']';
    dst[pos] = '\0';
    return 0;
}

int cloud_client_send_split(
    transport_client_t *transport,
    int split_id,
    const float *activation,
    size_t activation_len,
    const int *shape,
    size_t shape_len,
    int use_quantization,
    const char *model_version,
    cloud_infer_result_t *out,
    char *err,
    size_t err_size
)
{
    int8_t *qbuf = NULL;
    float scale = 1.0f;
    const uint8_t *payload_bytes;
    size_t payload_len;
    char *payload_b64 = NULL;
    size_t payload_b64_len = 0;
    char request_id[64];
    char shape_json[128];
    char *req_json = NULL;
    size_t req_cap = 32768;
    char *resp_json = NULL;
    int rc = -1;

    if (!transport || !activation || activation_len == 0 || !shape || shape_len == 0 || !model_version || !out) {
        set_error(err, err_size, "Invalid args to cloud_client_send_split");
        return -1;
    }
    if (append_shape_json(shape_json, sizeof(shape_json), shape, shape_len) != 0) {
        set_error(err, err_size, "Failed to encode shape JSON");
        return -1;
    }

    memset(out, 0, sizeof(*out));
    if (use_quantization) {
        qbuf = (int8_t *) malloc(sizeof(int8_t) * activation_len);
        if (!qbuf) {
            set_error(err, err_size, "Out of memory for quantization");
            return -1;
        }
        if (quantize_int8_symmetric(activation, activation_len, qbuf, &scale) != 0) {
            free(qbuf);
            set_error(err, err_size, "Quantization failed");
            return -1;
        }
        payload_bytes = (const uint8_t *) qbuf;
        payload_len = activation_len;
    } else {
        payload_bytes = (const uint8_t *) activation;
        payload_len = activation_len * sizeof(float);
    }

    if (base64_encode(payload_bytes, payload_len, &payload_b64, &payload_b64_len) != 0) {
        free(qbuf);
        set_error(err, err_size, "base64 encoding failed");
        return -1;
    }

    make_request_id(request_id, sizeof(request_id));
    req_json = (char *) malloc(req_cap);
    if (!req_json) {
        free(qbuf);
        free(payload_b64);
        set_error(err, err_size, "Out of memory");
        return -1;
    }

    if (use_quantization) {
        snprintf(
            req_json, req_cap,
            "{"
            "\"request_id\":\"%s\","
            "\"split_id\":%d,"
            "\"tensor_payload\":\"%s\","
            "\"shape\":%s,"
            "\"dtype\":\"int8\","
            "\"quantization_params\":{\"scale\":%.9g,\"zero_point\":0,\"dtype\":\"int8\"},"
            "\"model_version\":\"%s\","
            "\"trace_metadata\":{\"edge_id\":\"edge-native\"},"
            "\"edge_timestamp_ms\":%.3f"
            "}",
            request_id, split_id, payload_b64, shape_json, scale, model_version, now_ms()
        );
    } else {
        snprintf(
            req_json, req_cap,
            "{"
            "\"request_id\":\"%s\","
            "\"split_id\":%d,"
            "\"tensor_payload\":\"%s\","
            "\"shape\":%s,"
            "\"dtype\":\"float32\","
            "\"quantization_params\":null,"
            "\"model_version\":\"%s\","
            "\"trace_metadata\":{\"edge_id\":\"edge-native\"},"
            "\"edge_timestamp_ms\":%.3f"
            "}",
            request_id, split_id, payload_b64, shape_json, model_version, now_ms()
        );
    }

    if (transport_client_post_json(transport, "/infer/split", req_json, &resp_json) != 0) {
        set_error(err, err_size, "HTTP POST /infer/split failed");
        goto cleanup;
    }

    if (json_extract_string(resp_json, "status", out->status, sizeof(out->status)) != 0) {
        set_error(err, err_size, "Failed to parse response status");
        goto cleanup;
    }
    if (json_extract_int(resp_json, "predicted_class", &out->predicted_class) != 0) {
        set_error(err, err_size, "Failed to parse predicted_class");
        goto cleanup;
    }
    if (json_extract_string(resp_json, "predicted_label", out->predicted_label, sizeof(out->predicted_label)) != 0) {
        set_error(err, err_size, "Failed to parse predicted_label");
        goto cleanup;
    }
    if (json_extract_float(resp_json, "total_ms", &out->timing_total_ms) != 0) {
        set_error(err, err_size, "Failed to parse timing.total_ms");
        goto cleanup;
    }
    if (strcmp(out->status, "ok") != 0) {
        set_error(err, err_size, "Cloud response status != ok");
        goto cleanup;
    }

    rc = 0;

cleanup:
    free(qbuf);
    free(payload_b64);
    free(req_json);
    transport_response_free(resp_json);
    return rc;
}

int cloud_client_send_split_k7(
    transport_client_t *transport,
    const float activation[64],
    int use_quantization,
    const char *model_version,
    cloud_infer_result_t *out,
    char *err,
    size_t err_size
)
{
    static const int shape[] = {1, 64};
    return cloud_client_send_split(
        transport,
        7,
        activation,
        64,
        shape,
        2,
        use_quantization,
        model_version,
        out,
        err,
        err_size
    );
}
