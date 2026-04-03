#include "controller_linucb.h"

#include <math.h>
#include <string.h>

static void featurize(const controller_context_t *ctx, double phi[LINUCB_FEATURE_DIM])
{
    double norm;

    phi[0] = (double) ctx->rtt_ms;
    phi[1] = (double) ctx->cpu_util;
    phi[2] = (double) ctx->entropy;
    phi[3] = 1.0;

    norm = sqrt(phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2] + phi[3] * phi[3]);
    if (norm > 1e-10) {
        phi[0] /= norm;
        phi[1] /= norm;
        phi[2] /= norm;
        phi[3] /= norm;
    }
}

static void mat_vec_mul(
    const double m[LINUCB_FEATURE_DIM][LINUCB_FEATURE_DIM],
    const double v[LINUCB_FEATURE_DIM],
    double out[LINUCB_FEATURE_DIM]
)
{
    size_t i;
    size_t j;
    for (i = 0; i < LINUCB_FEATURE_DIM; i++) {
        out[i] = 0.0;
        for (j = 0; j < LINUCB_FEATURE_DIM; j++) {
            out[i] += m[i][j] * v[j];
        }
    }
}

static double vec_dot(const double *a, const double *b, size_t n)
{
    size_t i;
    double s = 0.0;
    for (i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

static int invert_4x4(
    const double in[LINUCB_FEATURE_DIM][LINUCB_FEATURE_DIM],
    double out[LINUCB_FEATURE_DIM][LINUCB_FEATURE_DIM]
)
{
    size_t i;
    size_t j;
    size_t k;
    double aug[LINUCB_FEATURE_DIM][LINUCB_FEATURE_DIM * 2];

    for (i = 0; i < LINUCB_FEATURE_DIM; i++) {
        for (j = 0; j < LINUCB_FEATURE_DIM; j++) {
            aug[i][j] = in[i][j];
            aug[i][j + LINUCB_FEATURE_DIM] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (i = 0; i < LINUCB_FEATURE_DIM; i++) {
        double pivot = aug[i][i];
        size_t pivot_row = i;
        if (fabs(pivot) < 1e-12) {
            for (j = i + 1; j < LINUCB_FEATURE_DIM; j++) {
                if (fabs(aug[j][i]) > fabs(pivot)) {
                    pivot = aug[j][i];
                    pivot_row = j;
                }
            }
            if (fabs(pivot) < 1e-12) {
                return -1;
            }
            if (pivot_row != i) {
                for (k = 0; k < LINUCB_FEATURE_DIM * 2; k++) {
                    double t = aug[i][k];
                    aug[i][k] = aug[pivot_row][k];
                    aug[pivot_row][k] = t;
                }
            }
        }

        pivot = aug[i][i];
        for (k = 0; k < LINUCB_FEATURE_DIM * 2; k++) {
            aug[i][k] /= pivot;
        }

        for (j = 0; j < LINUCB_FEATURE_DIM; j++) {
            if (j == i) {
                continue;
            }
            {
                double factor = aug[j][i];
                for (k = 0; k < LINUCB_FEATURE_DIM * 2; k++) {
                    aug[j][k] -= factor * aug[i][k];
                }
            }
        }
    }

    for (i = 0; i < LINUCB_FEATURE_DIM; i++) {
        for (j = 0; j < LINUCB_FEATURE_DIM; j++) {
            out[i][j] = aug[i][j + LINUCB_FEATURE_DIM];
        }
    }
    return 0;
}

int controller_linucb_init(
    controller_linucb_t *ctrl,
    const int *split_ids,
    size_t count,
    float alpha
)
{
    size_t arm_i;
    size_t i;

    if (!ctrl || feasible_split_set_init(&ctrl->feasible, split_ids, count) != 0) {
        return -1;
    }

    ctrl->alpha = alpha;
    memset(ctrl->A, 0, sizeof(ctrl->A));
    memset(ctrl->b, 0, sizeof(ctrl->b));
    memset(ctrl->updates, 0, sizeof(ctrl->updates));

    for (arm_i = 0; arm_i < ctrl->feasible.count; arm_i++) {
        for (i = 0; i < LINUCB_FEATURE_DIM; i++) {
            ctrl->A[arm_i][i][i] = 1.0;
        }
    }
    return 0;
}

void controller_linucb_reset(controller_linucb_t *ctrl)
{
    size_t arm_i;
    size_t i;

    if (!ctrl) {
        return;
    }

    memset(ctrl->A, 0, sizeof(ctrl->A));
    memset(ctrl->b, 0, sizeof(ctrl->b));
    memset(ctrl->updates, 0, sizeof(ctrl->updates));
    for (arm_i = 0; arm_i < ctrl->feasible.count; arm_i++) {
        for (i = 0; i < LINUCB_FEATURE_DIM; i++) {
            ctrl->A[arm_i][i][i] = 1.0;
        }
    }
}

int controller_linucb_select(
    const controller_linucb_t *ctrl,
    const controller_context_t *ctx,
    int *out_split_id
)
{
    size_t arm_i;
    double phi[LINUCB_FEATURE_DIM];
    int selected = -1;
    double best_ucb = -1e300;

    if (!ctrl || !ctx || !out_split_id || ctrl->feasible.count == 0) {
        return -1;
    }

    featurize(ctx, phi);
    for (arm_i = 0; arm_i < ctrl->feasible.count; arm_i++) {
        double A_inv[LINUCB_FEATURE_DIM][LINUCB_FEATURE_DIM];
        double theta[LINUCB_FEATURE_DIM];
        double tmp[LINUCB_FEATURE_DIM];
        double exploit;
        double explore;
        double ucb;
        double bonus_arg;

        if (invert_4x4(ctrl->A[arm_i], A_inv) != 0) {
            return -1;
        }

        mat_vec_mul(A_inv, ctrl->b[arm_i], theta);
        mat_vec_mul(A_inv, phi, tmp);
        exploit = vec_dot(theta, phi, LINUCB_FEATURE_DIM);
        bonus_arg = vec_dot(phi, tmp, LINUCB_FEATURE_DIM);
        if (bonus_arg < 0.0) {
            bonus_arg = 0.0;
        }
        explore = ctrl->alpha * sqrt(bonus_arg);
        ucb = exploit + explore;

        if (selected < 0 || ucb > best_ucb) {
            best_ucb = ucb;
            selected = ctrl->feasible.split_ids[arm_i];
        }
    }

    *out_split_id = selected;
    return 0;
}

int controller_linucb_update(
    controller_linucb_t *ctrl,
    const controller_context_t *ctx,
    int split_id,
    float reward
)
{
    int arm_index;
    size_t i;
    size_t j;
    double phi[LINUCB_FEATURE_DIM];

    if (!ctrl || !ctx) {
        return -1;
    }

    arm_index = feasible_split_set_index_of(&ctrl->feasible, split_id);
    if (arm_index < 0) {
        return -1;
    }

    featurize(ctx, phi);
    for (i = 0; i < LINUCB_FEATURE_DIM; i++) {
        for (j = 0; j < LINUCB_FEATURE_DIM; j++) {
            ctrl->A[arm_index][i][j] += phi[i] * phi[j];
        }
        ctrl->b[arm_index][i] += ((double) reward) * phi[i];
    }
    ctrl->updates[arm_index] += 1U;
    return 0;
}

int controller_linucb_get_arm_stats(
    const controller_linucb_t *ctrl,
    int split_id,
    controller_linucb_arm_stats_t *out_stats
)
{
    int arm_index;
    size_t i;
    double A_inv[LINUCB_FEATURE_DIM][LINUCB_FEATURE_DIM];
    double theta[LINUCB_FEATURE_DIM];

    if (!ctrl || !out_stats) {
        return -1;
    }

    arm_index = feasible_split_set_index_of(&ctrl->feasible, split_id);
    if (arm_index < 0) {
        return -1;
    }

    if (invert_4x4(ctrl->A[arm_index], A_inv) != 0) {
        return -1;
    }
    mat_vec_mul(A_inv, ctrl->b[arm_index], theta);

    out_stats->theta_norm = sqrt(vec_dot(theta, theta, LINUCB_FEATURE_DIM));
    out_stats->trace_A = 0.0;
    for (i = 0; i < LINUCB_FEATURE_DIM; i++) {
        out_stats->trace_A += ctrl->A[arm_index][i][i];
    }
    out_stats->updates = ctrl->updates[arm_index];
    return 0;
}
