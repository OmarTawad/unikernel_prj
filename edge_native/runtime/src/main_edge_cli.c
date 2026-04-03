#include "cloud_client.h"
#include "edge_model.h"
#include "edge_runtime.h"
#include "tensor.h"
#include "transport_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EDGE_OUTPUT_MAX (EDGE_OUT_CH2 * EDGE_OUT_LEN2)

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--split-id N] [--artifacts-root DIR] [--artifacts-dir DIR]\\n"
            "          [--input-bin FILE] [--dump-activation FILE]\\n"
            "          [--post] [--cloud-url URL] [--transport-backend NAME] [--transport-endpoint URI]\\n"
            "          [--cloud-path PATH] [--retries N] [--model-version VER] [--no-quant]\\n",
            prog);
}

static const char *env_or_default(const char *key, const char *fallback)
{
    const char *v = getenv(key);
    if (!v || v[0] == '\0') {
        return fallback;
    }
    return v;
}

int main(int argc, char **argv)
{
    int split_id = 7;
    const char *artifacts_root = env_or_default("UNISPLIT_ARTIFACTS_ROOT", "edge_native/artifacts/c_splits");
    const char *artifacts_dir = NULL;
    const char *input_bin = NULL;
    const char *dump_activation = NULL;
    const char *cloud_url = env_or_default("UNISPLIT_CLOUD_URL", "http://localhost:8000");
    const char *transport_backend = env_or_default("UNISPLIT_TRANSPORT_BACKEND", "posix");
    const char *transport_endpoint = NULL;
    const char *cloud_path = env_or_default("UNISPLIT_TRANSPORT_PATH", "/infer/split");
    const char *model_version = env_or_default("UNISPLIT_MODEL_VERSION", "v0.1.0");
    int do_post = 0;
    int use_quant = 1;
    int retries = 1;
    int i;

    char resolved_artifact_dir[512];
    char default_input_path[512];
    edge_model_t model;
    char err[256];
    float input[EDGE_INPUT_LEN] = {0};
    float output[EDGE_OUTPUT_MAX] = {0};
    size_t output_len = 0;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--split-id") == 0 && i + 1 < argc) {
            split_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--artifacts-root") == 0 && i + 1 < argc) {
            artifacts_root = argv[++i];
        } else if (strcmp(argv[i], "--artifacts-dir") == 0 && i + 1 < argc) {
            artifacts_dir = argv[++i];
        } else if (strcmp(argv[i], "--input-bin") == 0 && i + 1 < argc) {
            input_bin = argv[++i];
        } else if (strcmp(argv[i], "--dump-activation") == 0 && i + 1 < argc) {
            dump_activation = argv[++i];
        } else if (strcmp(argv[i], "--post") == 0) {
            do_post = 1;
        } else if (strcmp(argv[i], "--cloud-url") == 0 && i + 1 < argc) {
            cloud_url = argv[++i];
        } else if (strcmp(argv[i], "--transport-backend") == 0 && i + 1 < argc) {
            transport_backend = argv[++i];
        } else if (strcmp(argv[i], "--transport-endpoint") == 0 && i + 1 < argc) {
            transport_endpoint = argv[++i];
        } else if (strcmp(argv[i], "--cloud-path") == 0 && i + 1 < argc) {
            cloud_path = argv[++i];
        } else if (strcmp(argv[i], "--retries") == 0 && i + 1 < argc) {
            retries = atoi(argv[++i]);
            if (retries <= 0) {
                fprintf(stderr, "Invalid retries value\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--model-version") == 0 && i + 1 < argc) {
            model_version = argv[++i];
        } else if (strcmp(argv[i], "--no-quant") == 0) {
            use_quant = 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!edge_model_is_supported_split(split_id)) {
        fprintf(stderr, "Unsupported split id: %d\\n", split_id);
        return 1;
    }

    if (!artifacts_dir) {
        snprintf(resolved_artifact_dir, sizeof(resolved_artifact_dir), "%s/edge_k%d", artifacts_root, split_id);
        artifacts_dir = resolved_artifact_dir;
    }

    if (!input_bin) {
        snprintf(default_input_path, sizeof(default_input_path), "%s/reference_input.bin", artifacts_dir);
        input_bin = default_input_path;
    }

    if (!transport_endpoint) {
        transport_endpoint = getenv("UNISPLIT_TRANSPORT_ENDPOINT");
    }

    if (load_f32_file_exact(input_bin, input, EDGE_INPUT_LEN) != 0) {
        fprintf(stderr, "Failed to load input file: %s\\n", input_bin);
        return 1;
    }

    if (edge_model_load_from_dir(artifacts_dir, &model, err, sizeof(err)) != 0) {
        fprintf(stderr, "Model load failed: %s\\n", err);
        return 1;
    }
    if (model.split_id != split_id) {
        fprintf(stderr, "Artifact split mismatch: expected=%d found=%d\\n", split_id, model.split_id);
        edge_model_free(&model);
        return 1;
    }

    if (edge_runtime_forward(&model, input, output, EDGE_OUTPUT_MAX, &output_len, err, sizeof(err)) != 0) {
        fprintf(stderr, "Forward failed: %s\\n", err);
        edge_model_free(&model);
        return 1;
    }

    printf("EDGE_OK split=%d output_len=%zu activation0=%.6f\\n", split_id, output_len, output[0]);

    if (dump_activation && write_f32_file(dump_activation, output, output_len) != 0) {
        fprintf(stderr, "Failed to write activation output: %s\\n", dump_activation);
        edge_model_free(&model);
        return 1;
    }

    if (do_post) {
        transport_client_t transport;
        cloud_infer_result_t result;
        int req_shape[3];
        size_t req_shape_len = 0;
        if (split_id == 9) {
            printf("LOCAL_ONLY split=9 skips cloud request\\n");
        } else {
            const char *endpoint = transport_endpoint ? transport_endpoint : cloud_url;

            if (split_id == 0) {
                req_shape[0] = 1;
                req_shape[1] = 1;
                req_shape[2] = EDGE_INPUT_LEN;
                req_shape_len = 3;
            } else if (split_id == 3 || split_id == 6) {
                req_shape[0] = 1;
                req_shape[1] = model.output_shape[0];
                req_shape[2] = model.output_shape[1];
                req_shape_len = 3;
            } else {
                req_shape[0] = 1;
                req_shape[1] = model.output_shape[0];
                req_shape_len = 2;
            }

            if (transport_create_by_name(transport_backend, endpoint, 10, &transport, err, sizeof(err)) != 0) {
                fprintf(stderr, "Transport init failed: %s\\n", err);
                edge_model_free(&model);
                return 1;
            }
            printf("TRANSPORT_BACKEND=%s endpoint=%s\\n", transport_backend, endpoint);

            {
                int attempt;
                int success = 0;
                for (attempt = 1; attempt <= retries; attempt++) {
                    if (cloud_client_send_split_to_path(
                            &transport,
                            cloud_path,
                            split_id,
                            output,
                            output_len,
                            req_shape,
                            req_shape_len,
                            use_quant,
                            model_version,
                            &result,
                            err,
                            sizeof(err)) == 0) {
                        success = 1;
                        break;
                    }
                }
                if (!success) {
                    fprintf(stderr, "Cloud request failed: %s\\n", err);
                    transport_client_destroy(&transport);
                    edge_model_free(&model);
                    return 1;
                }
            }

            printf("CLOUD_OK split=%d status=%s class=%d label=%s total_ms=%.3f\\n",
                   split_id,
                   result.status,
                   result.predicted_class,
                   result.predicted_label,
                   result.timing_total_ms);
            transport_client_destroy(&transport);
        }
    }

    edge_model_free(&model);
    return 0;
}
