#include "cloud_client.h"

#include "base64.h"
#include "quantize.h"
#include "transport.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define K7_LEN 64

static void set_error(char *err, unsigned long err_size, const char *msg)
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
    snprintf(dst, size, "edge-k7-%ld-%ld", (long) ts.tv_sec, (long) ts.tv_nsec);
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

int cloud_client_send_split_k7(
    const transport_cfg_t *transport,
    const float activation[64],
    int use_quantization,
    const char *model_version,
    cloud_infer_result_t *out,
    char *err,
    unsigned long err_size
)
{
    int8_t qbuf[K7_LEN];
    float scale = 1.0f;
    const uint8_t *payload_bytes;
    size_t payload_len;
    char *payload_b64 = NULL;
    size_t payload_b64_len = 0;
    char request_id[64];
    char *req_json = NULL;
    size_t req_cap = 16384;
    char *resp_json = NULL;
    int rc = -1;

    if (!transport || !activation || !model_version || !out) {
        set_error(err, err_size, "Invalid arguments to cloud_client_send_split_k7");
        return -1;
    }

    memset(out, 0, sizeof(*out));

    if (use_quantization) {
        if (quantize_int8_symmetric(activation, K7_LEN, qbuf, &scale) != 0) {
            set_error(err, err_size, "Quantization failed");
            return -1;
        }
        payload_bytes = (const uint8_t *) qbuf;
        payload_len = K7_LEN;
    } else {
        payload_bytes = (const uint8_t *) activation;
        payload_len = K7_LEN * sizeof(float);
    }

    if (base64_encode(payload_bytes, payload_len, &payload_b64, &payload_b64_len) != 0) {
        set_error(err, err_size, "base64 encoding failed");
        return -1;
    }

    make_request_id(request_id, sizeof(request_id));

    req_json = (char *) malloc(req_cap);
    if (!req_json) {
        free(payload_b64);
        set_error(err, err_size, "Out of memory");
        return -1;
    }

    if (use_quantization) {
        snprintf(
            req_json,
            req_cap,
            "{"
            "\"request_id\":\"%s\"," 
            "\"split_id\":7,"
            "\"tensor_payload\":\"%s\"," 
            "\"shape\":[64],"
            "\"dtype\":\"int8\"," 
            "\"quantization_params\":{\"scale\":%.9g,\"zero_point\":0,\"dtype\":\"int8\"},"
            "\"model_version\":\"%s\"," 
            "\"trace_metadata\":{\"edge_id\":\"edge-native-k7\"},"
            "\"edge_timestamp_ms\":%.3f"
            "}",
            request_id,
            payload_b64,
            scale,
            model_version,
            now_ms()
        );
    } else {
        snprintf(
            req_json,
            req_cap,
            "{"
            "\"request_id\":\"%s\"," 
            "\"split_id\":7,"
            "\"tensor_payload\":\"%s\"," 
            "\"shape\":[64],"
            "\"dtype\":\"float32\"," 
            "\"quantization_params\":null,"
            "\"model_version\":\"%s\"," 
            "\"trace_metadata\":{\"edge_id\":\"edge-native-k7\"},"
            "\"edge_timestamp_ms\":%.3f"
            "}",
            request_id,
            payload_b64,
            model_version,
            now_ms()
        );
    }

    if (transport_post_json(transport, "/infer/split", req_json, &resp_json) != 0) {
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
    free(payload_b64);
    free(req_json);
    transport_free_response(resp_json);
    return rc;
}
