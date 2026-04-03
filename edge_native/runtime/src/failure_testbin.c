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
            "Usage:\\n"
            "  %s model-load <artifact_dir>\\n"
            "  %s forward-small-output <artifact_dir> <input_bin>\\n"
            "  %s cloud-invalid-shape\\n"
            "  %s cloud-ukstub-mode <ok|error|badjson>\\n"
            "  %s transport-posix-connect-fail\\n",
            prog, prog, prog, prog, prog);
}

int main(int argc, char **argv)
{
    char err[256] = {0};

    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    if (strcmp(argv[1], "model-load") == 0) {
        edge_model_t model;
        if (argc != 3) {
            usage(argv[0]);
            return 2;
        }
        if (edge_model_load_from_dir(argv[2], &model, err, sizeof(err)) != 0) {
            printf("EXPECTED_FAIL model-load err=%s\n", err);
            return 0;
        }
        edge_model_free(&model);
        fprintf(stderr, "Expected load failure but succeeded\n");
        return 1;
    }

    if (strcmp(argv[1], "forward-small-output") == 0) {
        edge_model_t model;
        float input[EDGE_INPUT_LEN];
        float out_small[4];
        size_t out_len = 0;
        if (argc != 4) {
            usage(argv[0]);
            return 2;
        }
        if (load_f32_file_exact(argv[3], input, EDGE_INPUT_LEN) != 0) {
            fprintf(stderr, "failed to load input\n");
            return 1;
        }
        if (edge_model_load_from_dir(argv[2], &model, err, sizeof(err)) != 0) {
            fprintf(stderr, "failed to load model: %s\n", err);
            return 1;
        }
        if (edge_runtime_forward(&model, input, out_small, 4, &out_len, err, sizeof(err)) == 0) {
            edge_model_free(&model);
            fprintf(stderr, "Expected forward-small-output to fail\n");
            return 1;
        }
        edge_model_free(&model);
        printf("EXPECTED_FAIL forward-small-output err=%s\n", err);
        return 0;
    }

    if (strcmp(argv[1], "cloud-invalid-shape") == 0) {
        transport_client_t t;
        cloud_infer_result_t result;
        float act[64] = {0};
        int shape[1] = {64};

        if (transport_ukstub_create("ukstub://ok", 1, &t, err, sizeof(err)) != 0) {
            fprintf(stderr, "ukstub create failed: %s\n", err);
            return 1;
        }
        if (cloud_client_send_split(&t, 7, act, 64, shape, 0, 1, "v0.1.0", &result, err, sizeof(err)) == 0) {
            transport_client_destroy(&t);
            fprintf(stderr, "Expected cloud-invalid-shape to fail\n");
            return 1;
        }
        transport_client_destroy(&t);
        printf("EXPECTED_FAIL cloud-invalid-shape err=%s\n", err);
        return 0;
    }

    if (strcmp(argv[1], "cloud-ukstub-mode") == 0) {
        transport_client_t t;
        cloud_infer_result_t result;
        float act[64] = {0};
        int shape[2] = {1, 64};
        const char *mode;
        char endpoint[64];

        if (argc != 3) {
            usage(argv[0]);
            return 2;
        }
        mode = argv[2];
        snprintf(endpoint, sizeof(endpoint), "ukstub://%s", mode);

        if (transport_ukstub_create(endpoint, 1, &t, err, sizeof(err)) != 0) {
            fprintf(stderr, "ukstub create failed: %s\n", err);
            return 1;
        }
        if (cloud_client_send_split(&t, 7, act, 64, shape, 2, 1, "v0.1.0", &result, err, sizeof(err)) == 0) {
            transport_client_destroy(&t);
            if (strcmp(mode, "ok") != 0) {
                fprintf(stderr, "Expected cloud-ukstub-mode %s to fail\n", mode);
                return 1;
            }
            printf("MODE_OK status=%s class=%d\n", result.status, result.predicted_class);
            return 0;
        }
        transport_client_destroy(&t);
        if (strcmp(mode, "ok") == 0) {
            fprintf(stderr, "Expected cloud-ukstub-mode ok to succeed but failed: %s\n", err);
            return 1;
        }
        printf("EXPECTED_FAIL cloud-ukstub-mode=%s err=%s\n", mode, err);
        return 0;
    }

    if (strcmp(argv[1], "transport-posix-connect-fail") == 0) {
        transport_client_t t;
        char *resp = NULL;
        if (transport_posix_create("http://127.0.0.1:65500", 1, &t, err, sizeof(err)) != 0) {
            fprintf(stderr, "transport create failed unexpectedly: %s\n", err);
            return 1;
        }
        if (transport_client_post_json(&t, "/infer/split", "{}", &resp) == 0) {
            transport_response_free(resp);
            transport_client_destroy(&t);
            fprintf(stderr, "Expected connect fail\n");
            return 1;
        }
        transport_client_destroy(&t);
        printf("EXPECTED_FAIL transport-posix-connect-fail\n");
        return 0;
    }

    usage(argv[0]);
    return 2;
}
