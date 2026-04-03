#ifndef UNISPLIT_CONTROLLER_H
#define UNISPLIT_CONTROLLER_H

#include <stddef.h>

#define CONTROLLER_MAX_SPLITS 8

typedef struct {
    float rtt_ms;
    float cpu_util;
    float entropy;
    float reserved0;
} controller_context_t;

typedef struct {
    int split_ids[CONTROLLER_MAX_SPLITS];
    size_t count;
} feasible_split_set_t;

int feasible_split_set_init(feasible_split_set_t *set, const int *split_ids, size_t count);
int feasible_split_set_contains(const feasible_split_set_t *set, int split_id);
int feasible_split_set_index_of(const feasible_split_set_t *set, int split_id);

#endif
