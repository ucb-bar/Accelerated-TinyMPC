#pragma once

#ifdef USE_RVV
#include "admm_rvv.hpp"
#else
#include "types.hpp"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

int tiny_solve(TinySolver *solver);
void tiny_init(TinySolver *solver);

void update_primal(TinySolver *solver);
void backward_pass_grad(TinySolver *solver);
void forward_pass(TinySolver *solver);
void update_slack(TinySolver *solver);
void update_dual(TinySolver *solver);
void update_linear_cost(TinySolver *solver);

#ifdef __cplusplus
}
#endif
