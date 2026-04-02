#include "tensor.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t tensor_numel_from_shape(int ndim, const int *shape)
{
    size_t n = 1;
    int i;

    if (!shape || ndim <= 0 || ndim > TENSOR_MAX_DIMS) {
        return 0;
    }

    for (i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            return 0;
        }
        n *= (size_t) shape[i];
    }

    return n;
}

static void tensor_compute_stride(tensor_f32_t *tensor)
{
    int i;

    tensor->stride[tensor->ndim - 1] = 1;
    for (i = tensor->ndim - 2; i >= 0; i--) {
        tensor->stride[i] = tensor->stride[i + 1] * tensor->shape[i + 1];
    }
}

int tensor_f32_init_view(tensor_f32_t *tensor, float *data, int ndim, const int *shape)
{
    int i;

    if (!tensor || !data || !shape || ndim <= 0 || ndim > TENSOR_MAX_DIMS) {
        return -1;
    }

    tensor->data = data;
    tensor->ndim = ndim;
    tensor->numel = tensor_numel_from_shape(ndim, shape);
    tensor->owns_data = 0;
    if (tensor->numel == 0) {
        return -1;
    }

    for (i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
    }
    for (i = ndim; i < TENSOR_MAX_DIMS; i++) {
        tensor->shape[i] = 1;
        tensor->stride[i] = 1;
    }

    tensor_compute_stride(tensor);
    return 0;
}

int tensor_f32_alloc(tensor_f32_t *tensor, int ndim, const int *shape)
{
    size_t numel;

    if (!tensor || !shape || ndim <= 0 || ndim > TENSOR_MAX_DIMS) {
        return -1;
    }

    numel = tensor_numel_from_shape(ndim, shape);
    if (numel == 0) {
        return -1;
    }

    memset(tensor, 0, sizeof(*tensor));
    tensor->data = (float *) calloc(numel, sizeof(float));
    if (!tensor->data) {
        return -1;
    }

    tensor->ndim = ndim;
    tensor->numel = numel;
    tensor->owns_data = 1;
    memcpy(tensor->shape, shape, (size_t) ndim * sizeof(int));
    tensor_compute_stride(tensor);
    return 0;
}

void tensor_f32_free(tensor_f32_t *tensor)
{
    if (!tensor) {
        return;
    }

    if (tensor->owns_data && tensor->data) {
        free(tensor->data);
    }

    memset(tensor, 0, sizeof(*tensor));
}

int load_f32_file_dynamic(const char *path, float **out_data, size_t *out_count)
{
    FILE *fp;
    long size_bytes;
    size_t read_elems;
    size_t count;
    float *data;

    if (!path || !out_data || !out_count) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }

    size_bytes = ftell(fp);
    if (size_bytes < 0 || (size_bytes % (long) sizeof(float)) != 0) {
        fclose(fp);
        return -1;
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }

    count = (size_t) size_bytes / sizeof(float);
    data = (float *) malloc(count * sizeof(float));
    if (!data) {
        fclose(fp);
        return -1;
    }

    read_elems = fread(data, sizeof(float), count, fp);
    fclose(fp);

    if (read_elems != count) {
        free(data);
        return -1;
    }

    *out_data = data;
    *out_count = count;
    return 0;
}

int load_f32_file_exact(const char *path, float *out_data, size_t expected_count)
{
    FILE *fp;
    size_t read_elems;

    if (!path || !out_data || expected_count == 0) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }

    read_elems = fread(out_data, sizeof(float), expected_count, fp);
    fclose(fp);

    if (read_elems != expected_count) {
        return -1;
    }

    return 0;
}

int write_f32_file(const char *path, const float *data, size_t count)
{
    FILE *fp;
    size_t written;

    if (!path || !data) {
        return -1;
    }

    fp = fopen(path, "wb");
    if (!fp) {
        return -1;
    }

    written = fwrite(data, sizeof(float), count, fp);
    fclose(fp);

    return (written == count) ? 0 : -1;
}

int write_i8_file(const char *path, const int8_t *data, size_t count)
{
    FILE *fp;
    size_t written;

    if (!path || !data) {
        return -1;
    }

    fp = fopen(path, "wb");
    if (!fp) {
        return -1;
    }

    written = fwrite(data, sizeof(int8_t), count, fp);
    fclose(fp);

    return (written == count) ? 0 : -1;
}
