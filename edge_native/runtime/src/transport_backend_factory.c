#include "transport_backend.h"

#include <stdio.h>
#include <string.h>

int transport_create_by_name(
    const char *backend_name,
    const char *endpoint,
    int timeout_seconds,
    transport_client_t *out_client,
    char *err,
    size_t err_size
)
{
    if (!backend_name || !out_client) {
        if (err && err_size > 0) {
            snprintf(err, err_size, "Invalid arguments to transport_create_by_name");
        }
        return -1;
    }

    if (strcmp(backend_name, "ukstub") == 0) {
        return transport_ukstub_create(endpoint, timeout_seconds, out_client, err, err_size);
    }

    if (strcmp(backend_name, "lwip") == 0) {
        return transport_lwip_create(endpoint, timeout_seconds, out_client, err, err_size);
    }

    if (strcmp(backend_name, "posix") == 0) {
#ifdef UNISPLIT_NO_POSIX_BACKEND
        (void) endpoint;
        (void) timeout_seconds;
        if (err && err_size > 0) {
            snprintf(err, err_size, "posix backend unavailable in this build");
        }
        return -1;
#else
        return transport_posix_create(endpoint, timeout_seconds, out_client, err, err_size);
#endif
    }

    if (err && err_size > 0) {
        snprintf(err, err_size, "Unsupported transport backend: %s", backend_name);
    }
    return -1;
}
