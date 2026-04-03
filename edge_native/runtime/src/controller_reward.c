#include "controller_reward.h"

float controller_compute_reward(int prediction_correct, float latency_ms, float lambda_latency)
{
    float accuracy_term = prediction_correct ? 1.0f : 0.0f;
    float penalty = lambda_latency * latency_ms;
    return accuracy_term - penalty;
}
