#include "controller.h"

#include "edge_model.h"

int feasible_split_set_init(feasible_split_set_t *set, const int *split_ids, size_t count)
{
    size_t i;
    size_t j;

    if (!set || !split_ids || count == 0 || count > CONTROLLER_MAX_SPLITS) {
        return -1;
    }

    set->count = 0;
    for (i = 0; i < count; i++) {
        int split_id = split_ids[i];
        if (!edge_model_is_supported_split(split_id)) {
            return -1;
        }

        for (j = 0; j < set->count; j++) {
            if (set->split_ids[j] == split_id) {
                return -1;
            }
        }
        set->split_ids[set->count++] = split_id;
    }
    return 0;
}

int feasible_split_set_contains(const feasible_split_set_t *set, int split_id)
{
    return feasible_split_set_index_of(set, split_id) >= 0;
}

int feasible_split_set_index_of(const feasible_split_set_t *set, int split_id)
{
    size_t i;

    if (!set) {
        return -1;
    }

    for (i = 0; i < set->count; i++) {
        if (set->split_ids[i] == split_id) {
            return (int) i;
        }
    }
    return -1;
}
