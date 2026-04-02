#include "transport.h"

#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

static int connect_tcp(const char *host, int port, int timeout_seconds)
{
    struct addrinfo hints;
    struct addrinfo *res = NULL;
    struct addrinfo *it;
    int sock = -1;
    char port_str[16];

    snprintf(port_str, sizeof(port_str), "%d", port);
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host, port_str, &hints, &res) != 0) {
        return -1;
    }

    for (it = res; it != NULL; it = it->ai_next) {
        struct timeval tv;

        sock = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (sock < 0) {
            continue;
        }

        tv.tv_sec = timeout_seconds;
        tv.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        if (connect(sock, it->ai_addr, it->ai_addrlen) == 0) {
            break;
        }

        close(sock);
        sock = -1;
    }

    freeaddrinfo(res);
    return sock;
}

static int send_all(int sock, const char *buf, size_t len)
{
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(sock, buf + sent, len - sent, 0);
        if (n <= 0) {
            return -1;
        }
        sent += (size_t) n;
    }
    return 0;
}

int transport_post_json(
    const transport_cfg_t *cfg,
    const char *path,
    const char *req_json,
    char **resp_json_out
)
{
    int sock;
    size_t req_len;
    size_t req_cap;
    char *request;
    char *response = NULL;
    size_t response_len = 0;
    size_t response_cap = 0;
    char read_buf[4096];
    int status_code = 0;
    char *body;
    char *headers_end;

    if (!cfg || !path || !req_json || !resp_json_out) {
        return -1;
    }

    sock = connect_tcp(cfg->host, cfg->port, cfg->timeout_seconds);
    if (sock < 0) {
        return -1;
    }

    req_len = strlen(req_json);
    req_cap = req_len + 512;
    request = (char *) malloc(req_cap);
    if (!request) {
        close(sock);
        return -1;
    }

    snprintf(
        request,
        req_cap,
        "POST %s HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: application/json\r\n"
        "Connection: close\r\n"
        "Content-Length: %zu\r\n"
        "\r\n"
        "%s",
        path,
        cfg->host,
        cfg->port,
        req_len,
        req_json
    );

    if (send_all(sock, request, strlen(request)) != 0) {
        free(request);
        close(sock);
        return -1;
    }

    free(request);

    while (1) {
        ssize_t n = recv(sock, read_buf, sizeof(read_buf), 0);
        if (n == 0) {
            break;
        }
        if (n < 0) {
            free(response);
            close(sock);
            return -1;
        }

        if (response_len + (size_t) n + 1 > response_cap) {
            size_t new_cap = (response_cap == 0) ? 8192 : response_cap * 2;
            while (new_cap < response_len + (size_t) n + 1) {
                new_cap *= 2;
            }
            response = (char *) realloc(response, new_cap);
            if (!response) {
                close(sock);
                return -1;
            }
            response_cap = new_cap;
        }

        memcpy(response + response_len, read_buf, (size_t) n);
        response_len += (size_t) n;
    }

    close(sock);

    if (!response || response_len == 0) {
        free(response);
        return -1;
    }

    response[response_len] = '\0';

    if (sscanf(response, "HTTP/%*d.%*d %d", &status_code) != 1) {
        free(response);
        return -1;
    }

    headers_end = strstr(response, "\r\n\r\n");
    if (!headers_end) {
        free(response);
        return -1;
    }

    body = headers_end + 4;
    body = strdup(body);
    free(response);

    if (!body) {
        return -1;
    }

    if (status_code < 200 || status_code >= 300) {
        free(body);
        return -1;
    }

    *resp_json_out = body;
    return 0;
}

void transport_free_response(char *resp_json)
{
    free(resp_json);
}
