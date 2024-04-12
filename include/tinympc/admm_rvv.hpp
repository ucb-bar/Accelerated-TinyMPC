//
// Created by widyadewi on 2/23/24.
//

#pragma once
#ifndef TINYMPC_ADMM_RVV_HPP
#define TINYMPC_ADMM_RVV_HPP

#include "types_rvv.hpp"

#ifndef USE_MATVEC
#define USE_MATVEC 1
#endif

extern "C" {

static uint64_t startTimestamp;
#ifdef MEASURE_CYCLES
std::ofstream outputFile("cycle_output.csv");
#define CYCLE_CNT_WRAPPER(func, arg, name) \
    do { \
        struct timespec start, end; \
        clock_gettime(CLOCK_MONOTONIC, &start); \
        func(arg); \
        clock_gettime(CLOCK_MONOTONIC, &end); \
        uint64_t timediff = (end.tv_sec - start.tv_sec)* 1e9 + (end.tv_nsec - start.tv_nsec); \
        outputFile << name << ", " << timediff << std::endl; \
    } while(0)
#else
#define CYCLE_CNT_WRAPPER(func, arg, name) func(arg)
#endif

// u1 = x[:, i] * Kinf; u2 = u1 + d; u[:, i] = -u2
inline void forward_pass_1(TinySolver *solver, int i) {
#ifdef USE_MATVEC
    matvec(solver->cache->Kinf.array, solver->work->x.col(i), solver->work->u1.array, NINPUTS, NSTATES);
#else
    matmul(solver->work->x.col(i), solver->cache->Kinf.array, solver->work->u1.array, 1, NINPUTS, NSTATES);
#endif
    matadd(solver->work->u1.array, solver->work->d.col(i), solver->work->u2.array, 1, NINPUTS);
    matneg(solver->work->u2.array, solver->work->u.col(i), 1, NINPUTS);
}

// x[:, i+1] = Adyn * x[:, i] + Bdyn * u[:, i]
inline void forward_pass_2(TinySolver *solver, int i) {
#ifdef USE_MATVEC
    matvec(solver->work->Adyn.array, solver->work->x.col(i), solver->work->x1.array, NSTATES, NSTATES);
    matvec(solver->work->Bdyn.array, solver->work->u.col(i), solver->work->x2.array, NSTATES, NINPUTS);
#else
    matmul(solver->work->x.col(i), solver->work->Adyn.array, solver->work->x1.array, 1, NSTATES, NSTATES);
    matmul(solver->work->u.col(i), solver->work->Bdyn.array, solver->work->x2.array, 1, NSTATES, NINPUTS);
#endif
    matadd(solver->work->x1.array, solver->work->x2.array, solver->work->x.col(i + 1), 1, NSTATES);
}

// d[:, i] = Quu_inv * (BdynT * p[:, i+1] + r[:, i]);
inline void backward_pass_1(TinySolver *solver, int i) {
#ifdef USE_MATVEC
    matvec(solver->work->BdynT.array, solver->work->p.col(i + 1), solver->work->u1.array, NINPUTS, NSTATES);
    matadd(solver->work->r.col(i), solver->work->u1.array, solver->work->u2.array, 1, NINPUTS);
    matvec(solver->cache->Quu_inv.array, solver->work->u2.array, solver->work->d.col(i), NINPUTS, NINPUTS);
#else
    matmul(solver->work->p.col(i + 1), solver->work->BdynT.array, solver->work->u1.array, 1, NINPUTS, NSTATES);
    matadd(solver->work->r.col(i), solver->work->u1.array, solver->work->u2.array, 1, NINPUTS);
    matmul(solver->work->u2.array, solver->cache->Quu_inv.array, solver->work->d.col(i), 1, NINPUTS, NINPUTS);
#endif
}

// p[:, i] = q[:, i] + AmBKt * p[:, i + 1] - KinfT * r[:, i]
inline void backward_pass_2(TinySolver *solver, int i) {
#ifdef USE_MATVEC
    matvec(solver->cache->AmBKt.array, solver->work->p.col(i + 1), solver->work->x1.array, NSTATES, NSTATES);
    matvec(solver->cache->KinfT.array, solver->work->r.col(i), solver->work->x2.array, NSTATES, NINPUTS);
#else
    matmul(solver->work->p.col(i + 1), solver->cache->AmBKt.array, solver->work->x1.array, 1, NSTATES, NSTATES);
    matmul(solver->work->r.col(i), solver->cache->KinfT.array, solver->work->x2.array, 1, NSTATES, NINPUTS);
#endif
    matsub(solver->work->x1.array, solver->work->x2.array, solver->work->x3.array, 1, NSTATES);
    matadd(solver->work->x3.array, solver->work->q.col(i), solver->work->p.col(i), 1, NSTATES);
}

// y u znew  g x vnew
inline void update_dual_1(TinySolver *solver) {
    matadd(solver->work->y.array, solver->work->u.array, solver->work->m1.array, NHORIZON - 1, NINPUTS);
    matsub(solver->work->m1.array, solver->work->znew.array, solver->work->y.array, NHORIZON - 1, NINPUTS);
    matadd(solver->work->g.array, solver->work->x.array, solver->work->s1.array, NHORIZON, NSTATES);
    matsub(solver->work->s1.array, solver->work->vnew.array, solver->work->g.array, NHORIZON, NSTATES);
}

// Box constraints on input
inline void update_slack_1(TinySolver *solver) {
    matadd(solver->work->u.array, solver->work->y.array, solver->work->znew.array, NHORIZON - 1, NINPUTS);
    if (solver->settings->en_input_bound) {
        cwisemax(solver->work->u_min.array, solver->work->znew.array, solver->work->m1.array, NHORIZON - 1, NINPUTS);
        cwisemin(solver->work->u_max.array, solver->work->m1.array, solver->work->znew.array, NHORIZON - 1, NINPUTS);
    }
}

// Box constraints on state
inline void update_slack_2(TinySolver *solver) {
    matadd(solver->work->x.array, solver->work->g.array, solver->work->vnew.array, NHORIZON, NSTATES);
    if (solver->settings->en_state_bound) {
        cwisemax(solver->work->x_min.array, solver->work->vnew.array, solver->work->s1.array, NHORIZON, NSTATES);
        cwisemin(solver->work->x_max.array, solver->work->s1.array, solver->work->vnew.array, NHORIZON, NSTATES);
    }
}

inline void primal_residual_state(TinySolver *solver) {
    matsub(solver->work->x.array, solver->work->vnew.array, solver->work->s1.array, NHORIZON, NSTATES);
    cwiseabs(solver->work->s1.array, solver->work->s2.array, NHORIZON, NSTATES);
    solver->work->primal_residual_state = maxcoeff(solver->work->s2.array, NHORIZON, NSTATES);
}

inline void dual_residual_state(TinySolver *solver) {
    matsub(solver->work->v.array, solver->work->vnew.array, solver->work->s1.array, NHORIZON, NSTATES);
    cwiseabs(solver->work->s1.array, solver->work->s2.array, NHORIZON, NSTATES);
    solver->work->dual_residual_state = maxcoeff(solver->work->s2.array, NHORIZON, NSTATES) * solver->cache->rho;
}

inline void primal_residual_input(TinySolver *solver) {
    matsub(solver->work->u.array, solver->work->znew.array, solver->work->m1.array, NHORIZON - 1, NINPUTS);
    cwiseabs(solver->work->m1.array, solver->work->m2.array, NHORIZON - 1, NINPUTS);
    solver->work->primal_residual_input = maxcoeff(solver->work->m2.array, NHORIZON - 1, NINPUTS);
}

inline void dual_residual_input(TinySolver *solver) {
    matsub(solver->work->z.array, solver->work->znew.array, solver->work->m1.array, NHORIZON - 1, NINPUTS);
    cwiseabs(solver->work->m1.array, solver->work->m2.array, NHORIZON - 1, NINPUTS);
    solver->work->dual_residual_input = maxcoeff(solver->work->m2.array, NHORIZON - 1, NINPUTS) * solver->cache->rho;
}

inline void update_linear_cost_1(TinySolver *solver) {
    matsub(solver->work->znew.array, solver->work->y.array, solver->work->m1.array, NHORIZON - 1, NINPUTS);
    matmulf(solver->work->m1.array, solver->work->r.array, -solver->cache->rho, NHORIZON - 1, NINPUTS);
}

inline void update_linear_cost_2(TinySolver *solver, int i) {
    cwisemul(solver->work->Xref.col(i), solver->work->Q.array, solver->work->x1.array, 1, NSTATES);
    matneg(solver->work->x1.array, solver->work->q.col(i), 1, NSTATES);
}

inline void update_linear_cost_3(TinySolver *solver) {
    matsub(solver->work->vnew.array, solver->work->g.array, solver->work->s1.array, NHORIZON, NSTATES);
    matmulf(solver->work->s1.array, solver->work->s2.array, solver->cache->rho, NHORIZON, NSTATES);
    matsub(solver->work->q.array, solver->work->s2.array, solver->work->s1.array, NHORIZON, NSTATES);
    solver->work->q.set(solver->work->s1._data);
}

inline void update_linear_cost_4(TinySolver *solver) {
    matsub(solver->work->vnew.col(NHORIZON - 1), solver->work->g.col(NHORIZON - 1), solver->work->x1.array, 1, NSTATES);
    matmulf(solver->work->x1.array, solver->work->x2.array, solver->cache->rho, 1, NSTATES);
#ifdef USE_MATVEC
    matvec(solver->cache->PinfT.array, solver->work->Xref.col(NHORIZON - 1), solver->work->x1.array, NSTATES, NSTATES);
#else
    matmul(solver->work->Xref.col(NHORIZON - 1), solver->cache->PinfT.array, solver->work->x1.array, 1, NSTATES, NSTATES);
#endif
    matadd(solver->work->x1.array, solver->work->x2.array, solver->work->x3.array, 1, NSTATES);
    matneg(solver->work->x3.array, solver->work->p.col(NHORIZON - 1), 1, NSTATES);
}

/**
 * Update linear terms from Riccati backward pass
 */
inline void backward_pass(TinySolver *solver)
{
    for (int i = NHORIZON - 2; i >= 0; i--) {
        backward_pass_1(solver, i);
        backward_pass_2(solver, i);
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
 */
inline void forward_pass(TinySolver *solver)
{
    for (int i = 0; i < NHORIZON - 1; i++) {
        forward_pass_1(solver, i);
        forward_pass_2(solver, i);
    }
}

/**
 * Do backward Riccati pass then forward roll out
 */
inline void update_primal(TinySolver *solver)
{
    CYCLE_CNT_WRAPPER(backward_pass, solver, "update_primal_backward_pass");
    CYCLE_CNT_WRAPPER(forward_pass, solver, "update_primal_forward_pass");
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint projection function
 */
inline void update_slack(TinySolver *solver)
{
    update_slack_1(solver);
    update_slack_2(solver);
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
 */
inline void update_dual(TinySolver *solver)
{
    update_dual_1(solver);
}

/**
 * Update linear control cost terms in the Riccati feedback using
 * the changing slack and dual variables from ADMM
 */
inline void update_linear_cost(TinySolver *solver)
{
    update_linear_cost_1(solver);
    for (int i = 0; i < NHORIZON; i++) {
        update_linear_cost_2(solver, i);
    }
    update_linear_cost_3(solver);
    update_linear_cost_4(solver);
}

inline void tiny_init(TinySolver *solver) {

}

};
#endif //TINYMPC_ADMM_RVV_HPP
