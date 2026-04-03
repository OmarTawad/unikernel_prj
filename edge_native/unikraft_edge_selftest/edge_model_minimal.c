#include "edge_model.h"

int edge_model_is_supported_split(int split_id)
{
    switch (split_id) {
    case 0:
    case 3:
    case 6:
    case 7:
    case 8:
    case 9:
        return 1;
    default:
        return 0;
    }
}

int edge_model_output_shape_for_split(int split_id, int *shape_out, size_t *ndim_out, size_t *len_out)
{
    if (!shape_out || !ndim_out || !len_out) {
        return -1;
    }

    shape_out[0] = 1;
    shape_out[1] = 1;
    shape_out[2] = 1;

    switch (split_id) {
    case 0:
        *ndim_out = 2;
        shape_out[0] = 1;
        shape_out[1] = EDGE_INPUT_LEN;
        *len_out = EDGE_INPUT_LEN;
        return 0;
    case 3:
        *ndim_out = 2;
        shape_out[0] = EDGE_OUT_CH1;
        shape_out[1] = EDGE_OUT_LEN1;
        *len_out = EDGE_OUT_CH1 * EDGE_OUT_LEN1;
        return 0;
    case 6:
        *ndim_out = 2;
        shape_out[0] = EDGE_OUT_CH2;
        shape_out[1] = EDGE_OUT_LEN2;
        *len_out = EDGE_OUT_CH2 * EDGE_OUT_LEN2;
        return 0;
    case 7:
        *ndim_out = 1;
        shape_out[0] = EDGE_POOL_LEN;
        *len_out = EDGE_POOL_LEN;
        return 0;
    case 8:
        *ndim_out = 1;
        shape_out[0] = EDGE_FC1_LEN;
        *len_out = EDGE_FC1_LEN;
        return 0;
    case 9:
        *ndim_out = 1;
        shape_out[0] = EDGE_LOGITS_LEN;
        *len_out = EDGE_LOGITS_LEN;
        return 0;
    default:
        return -1;
    }
}
