#include "transport_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int mode;
} transport_ukstub_impl_t;

enum {
    UKSTUB_MODE_OK = 0,
    UKSTUB_MODE_ERROR = 1,
    UKSTUB_MODE_BADJSON = 2,
};

static void set_error(char *err, size_t err_size, const char *msg)
{
    if (err && err_size > 0) {
        snprintf(err, err_size, "%s", msg);
    }
}

static int parse_mode(const char *endpoint)
{
    if (!endpoint || endpoint[0] == '\0') {
        return UKSTUB_MODE_OK;
    }
    if (strstr(endpoint, "error")) {
        return UKSTUB_MODE_ERROR;
    }
    if (strstr(endpoint, "badjson")) {
        return UKSTUB_MODE_BADJSON;
    }
    return UKSTUB_MODE_OK;
}

static int ukstub_post_json(
    transport_client_t *client,
    const char *path,
    const char *req_json,
    char **resp_json_out
)
{
    transport_ukstub_impl_t *impl;
    char *resp = NULL;

    (void) path;
    (void) req_json;

    if (!client || !resp_json_out) {
        return -1;
    }

    impl = (transport_ukstub_impl_t *) client->impl;
    if (!impl) {
        return -1;
    }

    if (impl->mode == UKSTUB_MODE_BADJSON) {
        resp = strdup("{\"status\":\"ok\"");
    } else if (impl->mode == UKSTUB_MODE_ERROR) {
        resp = strdup(
            "{"
            "\"status\":\"error\","
            "\"predicted_class\":-1,"
            "\"predicted_label\":\"error\","
            "\"timing\":{\"total_ms\":0.123},"
            "\"error\":\"ukstub forced error\""
            "}"
        );
    } else {
        resp = strdup(
            "{"
            "\"status\":\"ok\","
            "\"predicted_class\":7,"
            "\"predicted_label\":\"ukstub_class\","
            "\"timing\":{\"total_ms\":0.321}"
            "}"
        );
    }

    if (!resp) {
        return -1;
    }

    *resp_json_out = resp;
    return 0;
}

static void ukstub_destroy(transport_client_t *client)
{
    if (!client) {
        return;
    }
    free(client->impl);
    client->impl = NULL;
    client->post_json = NULL;
    client->destroy = NULL;
}

int transport_ukstub_create(
    const char *endpoint,
    int timeout_seconds,
    transport_client_t *out_client,
    char *err,
    size_t err_size
)
{
    transport_ukstub_impl_t *impl;

    (void) timeout_seconds;

    if (!out_client) {
        return -1;
    }
    memset(out_client, 0, sizeof(*out_client));

    impl = (transport_ukstub_impl_t *) calloc(1, sizeof(*impl));
    if (!impl) {
        set_error(err, err_size, "Out of memory creating ukstub transport");
        return -1;
    }
    impl->mode = parse_mode(endpoint);

    out_client->impl = impl;
    out_client->post_json = ukstub_post_json;
    out_client->destroy = ukstub_destroy;
    return 0;
}
