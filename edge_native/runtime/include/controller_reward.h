#ifndef UNISPLIT_CONTROLLER_REWARD_H
#define UNISPLIT_CONTROLLER_REWARD_H

float controller_compute_reward(int prediction_correct, float latency_ms, float lambda_latency);

#endif
