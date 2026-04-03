#include "cloud_client.h"
#include "model_k7.h"
#include "tensor.h"
#include "transport_backend.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--artifacts-dir DIR] [--input-bin FILE] [--dump-activation FILE]\\n"
            "          [--post] [--cloud-url URL] [--transport-backend NAME] [--transport-endpoint URI]\\n"
            "          [--model-version VER] [--no-quant]\\n",
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
    const char *artifacts_dir = env_or_default("UNISPLIT_ARTIFACTS_DIR", "edge_native/artifacts/edge_k7_c");
    const char *input_bin = NULL;
    const char *dump_activation = NULL;
    const char *cloud_url = env_or_default("UNISPLIT_CLOUD_URL", "http://localhost:8000");
    const char *transport_backend = env_or_default("UNISPLIT_TRANSPORT_BACKEND", "posix");
    const char *transport_endpoint = NULL;
    const char *model_version = env_or_default("UNISPLIT_MODEL_VERSION", "v0.1.0");
    int do_post = 0;
    int use_quant = 1;
    int i;

    model_k7_params_t model;
    char err[256];
    float input[EDGE_K7_INPUT_LEN] = {0};
    float output[EDGE_K7_OUTPUT_LEN] = {0};

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--artifacts-dir") == 0 && i + 1 < argc) {
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
        } else if (strcmp(argv[i], "--model-version") == 0 && i + 1 < argc) {
            model_version = argv[++i];
        } else if (strcmp(argv[i], "--no-quant") == 0) {
            use_quant = 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!input_bin) {
        static char ref_path[512];
        snprintf(ref_path, sizeof(ref_path), "%s/reference_input.bin", artifacts_dir);
        input_bin = ref_path;
    }

    if (!transport_endpoint) {
        transport_endpoint = getenv("UNISPLIT_TRANSPORT_ENDPOINT");
    }

    if (load_f32_file_exact(input_bin, input, EDGE_K7_INPUT_LEN) != 0) {
        fprintf(stderr, "Failed to load input file: %s\\n", input_bin);
        return 1;
    }

    if (model_k7_load_from_dir(artifacts_dir, &model, err, sizeof(err)) != 0) {
        fprintf(stderr, "Model load failed: %s\\n", err);
        return 1;
    }

    if (model_k7_forward(&model, input, output, err, sizeof(err)) != 0) {
        fprintf(stderr, "Forward failed: %s\\n", err);
        model_k7_free(&model);
        return 1;
    }

    printf("EDGE_K7_OK activation0=%.6f activation63=%.6f\\n", output[0], output[63]);

    if (dump_activation && write_f32_file(dump_activation, output, EDGE_K7_OUTPUT_LEN) != 0) {
        fprintf(stderr, "Failed to write activation output: %s\\n", dump_activation);
        model_k7_free(&model);
        return 1;
    }

    if (do_post) {
        transport_client_t transport;
        cloud_infer_result_t result;
        int created = 0;
        const char *endpoint = transport_endpoint ? transport_endpoint : cloud_url;

        if (transport_create_by_name(transport_backend, endpoint, 10, &transport, err, sizeof(err)) != 0) {
            fprintf(stderr, "Transport init failed: %s\\n", err);
            model_k7_free(&model);
            return 1;
        }
        created = 1;
        printf("TRANSPORT_BACKEND=%s endpoint=%s\\n", transport_backend, endpoint);

        if (cloud_client_send_split_k7(
                &transport,
                output,
                use_quant,
                model_version,
                &result,
                err,
                sizeof(err)) != 0) {
            fprintf(stderr, "Cloud request failed: %s\\n", err);
            if (created) {
                transport_client_destroy(&transport);
            }
            model_k7_free(&model);
            return 1;
        }

        printf("CLOUD_OK status=%s class=%d label=%s total_ms=%.3f\\n",
               result.status,
               result.predicted_class,
               result.predicted_label,
               result.timing_total_ms);
        if (created) {
            transport_client_destroy(&transport);
        }
    }

    model_k7_free(&model);
    return 0;
}
