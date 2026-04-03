#ifndef UNISPLIT_TRANSPORT_BACKEND_H
#define UNISPLIT_TRANSPORT_BACKEND_H

#include <stddef.h>

typedef struct transport_client transport_client_t;

typedef int (*transport_post_json_fn)(
    transport_client_t *client,
    const char *path,
    const char *req_json,
    char **resp_json_out
);
typedef void (*transport_destroy_fn)(transport_client_t *client);

struct transport_client {
    transport_post_json_fn post_json;
    transport_destroy_fn destroy;
    void *impl;
};

int transport_client_post_json(
    transport_client_t *client,
    const char *path,
    const char *req_json,
    char **resp_json_out
);

void transport_client_destroy(transport_client_t *client);
void transport_response_free(char *resp_json);

int transport_posix_create(
    const char *base_url,
    int timeout_seconds,
    transport_client_t *out_client,
    char *err,
    size_t err_size
);

int transport_ukstub_create(
    const char *endpoint,
    int timeout_seconds,
    transport_client_t *out_client,
    char *err,
    size_t err_size
);

int transport_lwip_create(
    const char *endpoint,
    int timeout_seconds,
    transport_client_t *out_client,
    char *err,
    size_t err_size
);

int transport_create_by_name(
    const char *backend_name,
    const char *endpoint,
    int timeout_seconds,
    transport_client_t *out_client,
    char *err,
    size_t err_size
);

#endif
