#include "transport_backend.h"

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

typedef struct {
    char host[256];
    int port;
    int timeout_seconds;
} transport_posix_impl_t;

static void set_error(char *err, size_t err_size, const char *msg)
{
    if (err && err_size > 0) {
        snprintf(err, err_size, "%s", msg);
    }
}

static int parse_base_url(const char *base_url, char *host, size_t host_size, int *port)
{
    const char *prefix = "http://";
    const char *p;
    const char *slash;
    const char *colon = NULL;
    size_t host_len;

    if (!base_url || !host || !port) {
        return -1;
    }

    if (strncmp(base_url, prefix, strlen(prefix)) != 0) {
        return -1;
    }

    p = base_url + strlen(prefix);
    slash = strchr(p, '/');
    if (!slash) {
        slash = p + strlen(p);
    }

    {
        const char *it;
        for (it = p; it < slash; it++) {
            if (*it == ':') {
                colon = it;
            }
        }
    }

    if (colon) {
        host_len = (size_t) (colon - p);
        if (host_len == 0 || host_len >= host_size) {
            return -1;
        }
        memcpy(host, p, host_len);
        host[host_len] = '\0';
        *port = atoi(colon + 1);
        if (*port <= 0) {
            return -1;
        }
        return 0;
    }

    host_len = (size_t) (slash - p);
    if (host_len == 0 || host_len >= host_size) {
        return -1;
    }
    memcpy(host, p, host_len);
    host[host_len] = '\0';
    *port = 80;
    return 0;
}

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
        (void) setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        (void) setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

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

static int posix_post_json(
    transport_client_t *client,
    const char *path,
    const char *req_json,
    char **resp_json_out
)
{
    transport_posix_impl_t *impl;
    int sock;
    size_t req_len;
    size_t req_cap;
    char *request = NULL;
    char *response = NULL;
    size_t response_len = 0;
    size_t response_cap = 0;
    char read_buf[4096];
    int status_code = 0;
    char *headers_end;
    char *body = NULL;

    if (!client || !path || !req_json || !resp_json_out) {
        return -1;
    }

    impl = (transport_posix_impl_t *) client->impl;
    if (!impl) {
        return -1;
    }

    sock = connect_tcp(impl->host, impl->port, impl->timeout_seconds);
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
        impl->host,
        impl->port,
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
    if (status_code < 200 || status_code >= 300) {
        free(response);
        return -1;
    }

    headers_end = strstr(response, "\r\n\r\n");
    if (!headers_end) {
        free(response);
        return -1;
    }

    body = strdup(headers_end + 4);
    free(response);
    if (!body) {
        return -1;
    }

    *resp_json_out = body;
    return 0;
}

static void posix_destroy(transport_client_t *client)
{
    if (!client) {
        return;
    }
    free(client->impl);
    client->impl = NULL;
    client->post_json = NULL;
    client->destroy = NULL;
}

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

int transport_posix_create(
    const char *base_url,
    int timeout_seconds,
    transport_client_t *out_client,
    char *err,
    size_t err_size
)
{
    transport_posix_impl_t *impl;

    if (!out_client) {
        return -1;
    }
    memset(out_client, 0, sizeof(*out_client));

    impl = (transport_posix_impl_t *) calloc(1, sizeof(*impl));
    if (!impl) {
        set_error(err, err_size, "Out of memory creating transport");
        return -1;
    }

    if (parse_base_url(base_url, impl->host, sizeof(impl->host), &impl->port) != 0) {
        free(impl);
        set_error(err, err_size, "Invalid base_url (expected http://host:port)");
        return -1;
    }
    impl->timeout_seconds = timeout_seconds > 0 ? timeout_seconds : 10;

    out_client->impl = impl;
    out_client->post_json = posix_post_json;
    out_client->destroy = posix_destroy;
    return 0;
}
