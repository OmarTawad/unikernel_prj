#ifndef UNISPLIT_TRANSPORT_H
#define UNISPLIT_TRANSPORT_H

typedef struct {
    const char *host;
    int port;
    int timeout_seconds;
} transport_cfg_t;

/* Returns 0 on success, non-zero on failure. Caller must free response using
 * transport_free_response().
 */
int transport_post_json(
    const transport_cfg_t *cfg,
    const char *path,
    const char *req_json,
    char **resp_json_out
);

void transport_free_response(char *resp_json);

#endif
