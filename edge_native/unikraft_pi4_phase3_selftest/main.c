#include <stdio.h>

#include "edge_model.h"
#include "embedded_model.h"

#define PHASE3_SPLIT_ID 7

static void phase3_halt(void)
{
	for (;;)
		__asm__ volatile("wfe");
}

static int phase3_fail(const char *reason)
{
	printf("UNISPLIT_RPI4_P3_SELFTEST_FAIL:%s\n", reason);
	printf("UNISPLIT_RPI4_P3_SELFTEST_DONE\n");
	phase3_halt();
	return 1;
}

int main(void)
{
	edge_model_t model;
	char err[96];
	const float *reference_input;
	const char *strategy;
	int rc;

	printf("UNISPLIT_RPI4_P3_MAIN_ENTRY\n");
	printf("UNISPLIT_RPI4_P3_SELFTEST_START\n");

	rc = edge_model_load_embedded(&model, PHASE3_SPLIT_ID, err, sizeof(err));
	if (rc != 0)
		return phase3_fail("LOAD_RC");

	if (model.split_id != PHASE3_SPLIT_ID ||
	    model.output_ndim != 1 ||
	    model.output_len != EDGE_POOL_LEN ||
	    model.output_shape[0] != EDGE_POOL_LEN ||
	    model.output_shape[1] != 1 ||
	    model.output_shape[2] != 1)
		return phase3_fail("BAD_SHAPE");

	if (!model.conv1_weight || !model.bn1_gamma ||
	    !model.conv2_weight || !model.bn2_gamma ||
	    model.fc1_weight || model.fc2_weight)
		return phase3_fail("BAD_PTRS");

	reference_input = edge_embedded_reference_input();
	if (!reference_input || edge_embedded_reference_input_len() != EDGE_INPUT_LEN)
		return phase3_fail("BAD_REF_INPUT");

	strategy = edge_embedded_artifact_strategy();
	if (!strategy || strategy[0] == '\0')
		return phase3_fail("BAD_STRATEGY");

	printf("UNISPLIT_RPI4_P3_SELFTEST_PASS\n");
	printf("UNISPLIT_RPI4_P3_SELFTEST_DONE\n");
	phase3_halt();
	return 0;
}
