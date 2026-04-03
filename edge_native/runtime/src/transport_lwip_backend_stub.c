#include "transport_backend.h"

#include <stdio.h>
#include <string.h>

int transport_lwip_create(
    const char *endpoint,
    int timeout_seconds,
    transport_client_t *out_client,
    char *err,
    size_t err_size
)
{
    (void) endpoint;
    (void) timeout_seconds;

    if (out_client) {
        memset(out_client, 0, sizeof(*out_client));
    }

    if (err && err_size > 0) {
        snprintf(
            err,
            err_size,
            "lwip backend is not implemented yet (Pi hardware phase blocker)"
        );
    }
    return -1;
}
