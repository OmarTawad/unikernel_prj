#ifndef UNISPLIT_TENSOR_H
#define UNISPLIT_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#define TENSOR_MAX_DIMS 3

typedef struct {
    float *data;
    int ndim;
    int shape[TENSOR_MAX_DIMS];
    int stride[TENSOR_MAX_DIMS];
    size_t numel;
    int owns_data;
} tensor_f32_t;

size_t tensor_numel_from_shape(int ndim, const int *shape);
int tensor_f32_init_view(tensor_f32_t *tensor, float *data, int ndim, const int *shape);
int tensor_f32_alloc(tensor_f32_t *tensor, int ndim, const int *shape);
void tensor_f32_free(tensor_f32_t *tensor);

int load_f32_file_dynamic(const char *path, float **out_data, size_t *out_count);
int load_f32_file_exact(const char *path, float *out_data, size_t expected_count);
int write_f32_file(const char *path, const float *data, size_t count);
int write_i8_file(const char *path, const int8_t *data, size_t count);

#endif
