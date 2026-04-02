#include "cloud_client.h"
#include "model_k7.h"
#include "tensor.h"
#include "transport.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--artifacts-dir DIR] [--input-bin FILE] [--dump-activation FILE]\\n"
            "          [--post] [--cloud-url URL] [--model-version VER] [--no-quant]\\n",
            prog);
}

static int parse_http_url(const char *url, char *host, size_t host_size, int *port)
{
    const char *prefix = "http://";
    const char *p;
    const char *colon;

    if (!url || !host || !port) {
        return -1;
    }

    if (strncmp(url, prefix, strlen(prefix)) != 0) {
        return -1;
    }

    p = url + strlen(prefix);
    colon = strrchr(p, ':');
    if (!colon) {
        snprintf(host, host_size, "%s", p);
        *port = 80;
        return 0;
    }

    if ((size_t) (colon - p) >= host_size) {
        return -1;
    }

    memcpy(host, p, (size_t) (colon - p));
    host[colon - p] = '\0';
    *port = atoi(colon + 1);
    if (*port <= 0) {
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    const char *artifacts_dir = "edge_native/artifacts/edge_k7_c";
    const char *input_bin = NULL;
    const char *dump_activation = NULL;
    const char *cloud_url = "http://localhost:8000";
    const char *model_version = "v0.1.0";
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
        char host[256];
        int port = 0;
        transport_cfg_t transport;
        cloud_infer_result_t result;

        if (parse_http_url(cloud_url, host, sizeof(host), &port) != 0) {
            fprintf(stderr, "Invalid --cloud-url (expected http://host:port): %s\\n", cloud_url);
            model_k7_free(&model);
            return 1;
        }

        transport.host = host;
        transport.port = port;
        transport.timeout_seconds = 10;

        if (cloud_client_send_split_k7(
                &transport,
                output,
                use_quant,
                model_version,
                &result,
                err,
                sizeof(err)) != 0) {
            fprintf(stderr, "Cloud request failed: %s\\n", err);
            model_k7_free(&model);
            return 1;
        }

        printf("CLOUD_OK status=%s class=%d label=%s total_ms=%.3f\\n",
               result.status,
               result.predicted_class,
               result.predicted_label,
               result.timing_total_ms);
    }

    model_k7_free(&model);
    return 0;
}
