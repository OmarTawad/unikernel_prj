#include "controller_linucb.h"
#include "controller_reward.h"

#include <stdio.h>

int main(void)
{
    const int splits[] = {0, 3, 6, 7, 8, 9};
    controller_linucb_t ctrl;
    controller_context_t ctx = {20.0f, 0.6f, 1.2f, 0.0f};
    controller_context_t ctx2 = {60.0f, 0.8f, 1.8f, 0.0f};
    int selected = -1;
    controller_linucb_arm_stats_t stats_before;
    controller_linucb_arm_stats_t stats_after;
    float reward;
    int i;

    if (controller_linucb_init(&ctrl, splits, sizeof(splits) / sizeof(splits[0]), 1.0f) != 0) {
        fprintf(stderr, "init failed\\n");
        return 1;
    }

    if (controller_linucb_select(&ctrl, &ctx, &selected) != 0) {
        fprintf(stderr, "select failed\\n");
        return 1;
    }
    if (selected < 0) {
        fprintf(stderr, "invalid selected split\\n");
        return 1;
    }
    if (controller_linucb_get_arm_stats(&ctrl, selected, &stats_before) != 0) {
        fprintf(stderr, "stats_before failed\\n");
        return 1;
    }

    reward = controller_compute_reward(1, 12.5f, 0.01f);
    if (controller_linucb_update(&ctrl, &ctx, selected, reward) != 0) {
        fprintf(stderr, "update failed\\n");
        return 1;
    }
    if (controller_linucb_get_arm_stats(&ctrl, selected, &stats_after) != 0) {
        fprintf(stderr, "stats_after failed\\n");
        return 1;
    }
    if (stats_after.updates <= stats_before.updates) {
        fprintf(stderr, "stats updates did not increase\\n");
        return 1;
    }

    for (i = 0; i < 10; i++) {
        int arm = -1;
        float r = controller_compute_reward((i % 2) == 0, 15.0f + (float) i, 0.005f);
        if (controller_linucb_select(&ctrl, (i % 2) ? &ctx2 : &ctx, &arm) != 0) {
            fprintf(stderr, "loop select failed\\n");
            return 1;
        }
        if (controller_linucb_update(&ctrl, (i % 2) ? &ctx2 : &ctx, arm, r) != 0) {
            fprintf(stderr, "loop update failed\\n");
            return 1;
        }
    }

    controller_linucb_reset(&ctrl);
    if (controller_linucb_get_arm_stats(&ctrl, selected, &stats_after) != 0) {
        fprintf(stderr, "stats after reset failed\\n");
        return 1;
    }
    if (stats_after.updates != 0) {
        fprintf(stderr, "reset failed\\n");
        return 1;
    }

    printf("CONTROLLER_OK selected=%d reward=%.3f\\n", selected, reward);
    return 0;
}
