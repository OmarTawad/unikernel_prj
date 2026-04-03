#include "transport_backend.h"

#include <stdlib.h>

int transport_client_post_json(
    transport_client_t *client,
    const char *path,
    const char *req_json,
    char **resp_json_out
)
{
    if (!client || !client->post_json) {
        return -1;
    }
    return client->post_json(client, path, req_json, resp_json_out);
}

void transport_client_destroy(transport_client_t *client)
{
    if (client && client->destroy) {
        client->destroy(client);
    }
}

void transport_response_free(char *resp_json)
{
    free(resp_json);
}
