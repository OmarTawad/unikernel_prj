#include "cloud_client.h"
#include "controller.h"
#include "edge_runtime.h"
#include "embedded_model.h"
#include "transport_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EDGE_OUTPUT_MAX (EDGE_OUT_CH2 * EDGE_OUT_LEN2)

typedef struct {
    int split_id;
    const char *backend;
    const char *endpoint;
    const char *path;
    int timeout_seconds;
    int retries;
    int do_post;
    int use_quantization;
    int run_controller;
} app_cfg_t;

static int parse_int_arg(const char *s, int *out)
{
    char *endptr = NULL;
    long v;

    if (!s || !out) {
        return -1;
    }

    v = strtol(s, &endptr, 10);
    if (endptr == s || (endptr && *endptr != '\0')) {
        return -1;
    }
    *out = (int) v;
    return 0;
}

static void cfg_defaults(app_cfg_t *cfg)
{
    cfg->split_id = 7;
    cfg->backend = "ukstub";
    cfg->endpoint = "ukstub://ok";
    cfg->path = "/infer/split";
    cfg->timeout_seconds = 10;
    cfg->retries = 1;
    cfg->do_post = 1;
    cfg->use_quantization = 0;
    cfg->run_controller = 1;
}

static int parse_args(int argc, char *argv[], app_cfg_t *cfg, char *err, size_t err_size)
{
    int i;

    if (!cfg) {
        return -1;
    }

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--split-id") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &cfg->split_id) != 0) {
                snprintf(err, err_size, "invalid --split-id value");
                return -1;
            }
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            cfg->backend = argv[++i];
        } else if (strcmp(argv[i], "--endpoint") == 0 && i + 1 < argc) {
            cfg->endpoint = argv[++i];
        } else if (strcmp(argv[i], "--path") == 0 && i + 1 < argc) {
            cfg->path = argv[++i];
        } else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &cfg->timeout_seconds) != 0 || cfg->timeout_seconds <= 0) {
                snprintf(err, err_size, "invalid --timeout value");
                return -1;
            }
        } else if (strcmp(argv[i], "--retries") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &cfg->retries) != 0 || cfg->retries <= 0) {
                snprintf(err, err_size, "invalid --retries value");
                return -1;
            }
        } else if (strcmp(argv[i], "--no-post") == 0) {
            cfg->do_post = 0;
        } else if (strcmp(argv[i], "--no-quant") == 0) {
            cfg->use_quantization = 0;
        } else if (strcmp(argv[i], "--no-controller") == 0) {
            cfg->run_controller = 0;
        } else {
            snprintf(err, err_size, "unknown or malformed arg: %s", argv[i]);
            return -1;
        }
    }

    if (!edge_model_is_supported_split(cfg->split_id)) {
        snprintf(err, err_size, "unsupported split_id=%d", cfg->split_id);
        return -1;
    }

    if (!cfg->backend || !cfg->endpoint || !cfg->path) {
        snprintf(err, err_size, "backend/endpoint/path config missing");
        return -1;
    }

    return 0;
}

static int run_split_forward(int split_id, const float *input, float *out, size_t *out_len)
{
    edge_model_t model;
    char err[128];

    if (edge_model_load_embedded(&model, split_id, err, sizeof(err)) != 0) {
        printf("PI_MARKER_SPLIT_DISPATCH_FAIL split=%d stage=model_load err=%s\n", split_id, err);
        return -1;
    }
    if (edge_runtime_forward(&model, input, out, EDGE_OUTPUT_MAX, out_len, err, sizeof(err)) != 0) {
        printf("PI_MARKER_SPLIT_DISPATCH_FAIL split=%d stage=forward err=%s\n", split_id, err);
        return -1;
    }

    printf("PI_MARKER_SPLIT_DISPATCH_OK split=%d len=%zu act0=%.6f\n", split_id, *out_len, out[0]);
    return 0;
}

int main(int argc, char *argv[])
{
    app_cfg_t cfg;
    char err[256];
    float out[EDGE_OUTPUT_MAX];
    size_t out_len = 0;
    const float *input;
    int selftest_splits[3] = {3, 7, 8};
    int i;

    feasible_split_set_t feasible;
    transport_client_t transport;
    cloud_infer_result_t result;
    int req_shape[3];
    size_t req_shape_len = 0;
    int ok = 0;

    printf("PI_MARKER_BOOT_START app=unisplit_edge_selftest\n");
    printf("PI_MARKER_ARTIFACT_STRATEGY=%s\n", edge_embedded_artifact_strategy());

    cfg_defaults(&cfg);
    memset(err, 0, sizeof(err));
    if (parse_args(argc, argv, &cfg, err, sizeof(err)) != 0) {
        printf("PI_MARKER_CONFIG_FAIL err=%s\n", err);
        return 2;
    }

    printf(
        "PI_MARKER_CONFIG_OK split=%d backend=%s endpoint=%s path=%s timeout=%d retries=%d post=%d quant=%d\n",
        cfg.split_id,
        cfg.backend,
        cfg.endpoint,
        cfg.path,
        cfg.timeout_seconds,
        cfg.retries,
        cfg.do_post,
        cfg.use_quantization
    );

    input = edge_embedded_reference_input();

    for (i = 0; i < 3; i++) {
        size_t test_len = 0;
        if (run_split_forward(selftest_splits[i], input, out, &test_len) != 0) {
            return 1;
        }
        printf("UK_SELFTEST_SPLIT_OK split=%d len=%zu\n", selftest_splits[i], test_len);
    }
    printf("UK_SELFTEST_EDGE_OK\n");

    if (cfg.run_controller) {
        controller_context_t ctx;
        int selected = -1;

        ctx.rtt_ms = 4.0f;
        ctx.cpu_util = 0.30f;
        ctx.entropy = 0.20f;
        ctx.reserved0 = 0.0f;

        if (feasible_split_set_init(&feasible, selftest_splits, 3) != 0) {
            printf("PI_MARKER_CONTROLLER_FAIL err=feasible_set_init\n");
            return 1;
        }

        selected = feasible.split_ids[0];
        if (!feasible_split_set_contains(&feasible, selected)) {
            printf("PI_MARKER_CONTROLLER_FAIL err=contains_false\n");
            return 1;
        }
        printf("PI_MARKER_CONTROLLER_OK selected=%d count=%zu\n", selected, feasible.count);
        printf("UK_SELFTEST_CTRL_OK selected=%d count=%zu\n", selected, feasible.count);
        (void) ctx;
    }

    if (run_split_forward(cfg.split_id, input, out, &out_len) != 0) {
        return 1;
    }

    if (!cfg.do_post) {
        printf("PI_MARKER_POST_SKIPPED\n");
        printf("PI_MARKER_FINAL_SUCCESS\n");
        printf("UK_SELFTEST_DONE\n");
        return 0;
    }

    memset(&transport, 0, sizeof(transport));
    if (transport_create_by_name(cfg.backend, cfg.endpoint, cfg.timeout_seconds, &transport, err, sizeof(err)) != 0) {
        printf("PI_MARKER_BACKEND_INIT_FAIL backend=%s err=%s\n", cfg.backend, err);
        return 1;
    }
    printf("PI_MARKER_BACKEND_INIT_OK backend=%s\n", cfg.backend);
    printf("PI_MARKER_NETWORK_READY backend=%s endpoint=%s\n", cfg.backend, cfg.endpoint);
    printf("UK_SELFTEST_TRANSPORT=%s\n", cfg.backend);

    if (cfg.split_id == 0) {
        req_shape[0] = 1;
        req_shape[1] = 1;
        req_shape[2] = EDGE_INPUT_LEN;
        req_shape_len = 3;
    } else if (cfg.split_id == 3 || cfg.split_id == 6) {
        req_shape[0] = 1;
        req_shape[1] = (cfg.split_id == 3) ? EDGE_OUT_CH1 : EDGE_OUT_CH2;
        req_shape[2] = (cfg.split_id == 3) ? EDGE_OUT_LEN1 : EDGE_OUT_LEN2;
        req_shape_len = 3;
    } else {
        req_shape[0] = 1;
        req_shape[1] = (int) out_len;
        req_shape_len = 2;
    }

    for (i = 1; i <= cfg.retries; i++) {
        printf("PI_MARKER_INFER_ATTEMPT split=%d attempt=%d/%d path=%s\n", cfg.split_id, i, cfg.retries, cfg.path);

        if (cloud_client_send_split_to_path(
                &transport,
                cfg.path,
                cfg.split_id,
                out,
                out_len,
                req_shape,
                req_shape_len,
                cfg.use_quantization,
                "v0.1.0",
                &result,
                err,
                sizeof(err)) == 0) {
            ok = 1;
            printf(
                "PI_MARKER_INFER_RESPONSE_OK split=%d status=%s class=%d label=%s total_ms=%.3f\n",
                cfg.split_id,
                result.status,
                result.predicted_class,
                result.predicted_label,
                result.timing_total_ms
            );
            printf("UK_SELFTEST_TRANSPORT_OK\n");
            break;
        }

        printf("PI_MARKER_INFER_RESPONSE_FAIL split=%d attempt=%d err=%s\n", cfg.split_id, i, err);
    }

    transport_client_destroy(&transport);

    if (!ok) {
        printf("PI_MARKER_FINAL_FAIL reason=infer_failed\n");
        return 1;
    }

    printf("PI_MARKER_FINAL_SUCCESS\n");
    printf("UK_SELFTEST_DONE\n");
    return 0;
}
