#ifndef UNISPLIT_CONTROLLER_LINUCB_H
#define UNISPLIT_CONTROLLER_LINUCB_H

#include "controller.h"

#include <stddef.h>

#define LINUCB_FEATURE_DIM 4

typedef struct {
    feasible_split_set_t feasible;
    float alpha;
    double A[CONTROLLER_MAX_SPLITS][LINUCB_FEATURE_DIM][LINUCB_FEATURE_DIM];
    double b[CONTROLLER_MAX_SPLITS][LINUCB_FEATURE_DIM];
    unsigned int updates[CONTROLLER_MAX_SPLITS];
} controller_linucb_t;

typedef struct {
    double theta_norm;
    double trace_A;
    unsigned int updates;
} controller_linucb_arm_stats_t;

int controller_linucb_init(
    controller_linucb_t *ctrl,
    const int *split_ids,
    size_t count,
    float alpha
);

void controller_linucb_reset(controller_linucb_t *ctrl);

int controller_linucb_select(
    const controller_linucb_t *ctrl,
    const controller_context_t *ctx,
    int *out_split_id
);

int controller_linucb_update(
    controller_linucb_t *ctrl,
    const controller_context_t *ctx,
    int split_id,
    float reward
);

int controller_linucb_get_arm_stats(
    const controller_linucb_t *ctrl,
    int split_id,
    controller_linucb_arm_stats_t *out_stats
);

#endif
