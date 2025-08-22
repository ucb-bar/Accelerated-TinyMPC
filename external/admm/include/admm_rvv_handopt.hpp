//
// Created by widyadewi on 2/23/24.
//

#pragma once
#ifndef TINYMPC_ADMM_RVV_HANDOPT_HPP
#define TINYMPC_ADMM_RVV_HANDOPT_HPP

#include <types_rvv.hpp>

#ifndef USE_MATVEC
#define USE_MATVEC 1
#endif

extern "C" {

static uint64_t startTimestamp;

#define CYCLE_CNT_WRAPPER(func, arg, name) func(arg)


// // u1 = x[:, i] * Kinf; u2 = u1 + d; u[:, i] = -u2
// inline void forward_pass_1(TinySolver *solver, int i) {
// #ifdef USE_MATVEC
//     matvec(solver->cache->Kinf.data, solver->work->x.vector[i], solver->work->u1.data, NINPUTS, NSTATES);
// #else
//     matmul(solver->work->x.vector[i], solver->cache->Kinf.data, solver->work->u1.data, 1, NINPUTS, NSTATES);
// #endif
//     TRACE_CHECKSUM(forward_pass_1, solver->work->u1);
//     matadd(solver->work->u1.data, solver->work->d.vector[i], solver->work->u2.data, 1, NINPUTS);
//     TRACE_CHECKSUM(forward_pass_1, solver->work->u2);
//     matneg(solver->work->u2.data, solver->work->u.vector[i], 1, NINPUTS);
//     TRACE_CHECKSUM(forward_pass_1, solver->work->u);
// }

inline void forward_pass_1(TinySolver *solver, int i) {
    vfloat32m1_t vec_k_0, vec_k_1, vec_k_2, vec_k_3;
    vfloat32m1_t vec_b;
    vfloat32m1_t vec_s_0, vec_s_1, vec_s_2, vec_s_3, vec_s_4;
    vfloat32m1_t vec_sum_0, vec_sum_1, vec_sum_2, vec_sum_3;
    vfloat32m1_t vec_zero = __riscv_vfsub_vv_f32m1(vec_zero, vec_zero, NSTATES);


    float *ptr_k = solver->cache->KinfT.data;
    float *ptr_b = solver->work->x.vector[i];
    float *ptr_d = solver->work->d.vector[i];
    float ptr_c[4];
    float *ptr_u = solver->work->u.vector[i];
    float sum;

    vec_s_0 = vec_zero;
    vec_s_1 = vec_zero;
    vec_s_2 = vec_zero;
    vec_s_3 = vec_zero;

    // 0 - 3
    vec_k_0 = __riscv_vle32_v_f32m1(ptr_k + 0 * NINPUTS, NINPUTS);
    vec_k_1 = __riscv_vle32_v_f32m1(ptr_k + 1 * NINPUTS, NINPUTS);
    vec_k_2 = __riscv_vle32_v_f32m1(ptr_k + 2 * NINPUTS, NINPUTS);
    vec_k_3 = __riscv_vle32_v_f32m1(ptr_k + 3 * NINPUTS, NINPUTS);

    vec_s_0 = __riscv_vfmacc_vf_f32m1(vec_s_0, ptr_b[0], vec_k_0, NINPUTS);
    vec_s_1 = __riscv_vfmacc_vf_f32m1(vec_s_1, ptr_b[1], vec_k_1, NINPUTS);
    vec_s_2 = __riscv_vfmacc_vf_f32m1(vec_s_2, ptr_b[2], vec_k_2, NINPUTS);
    vec_s_3 = __riscv_vfmacc_vf_f32m1(vec_s_3, ptr_b[3], vec_k_3, NINPUTS);

    // 4 - 7
    vec_k_0 = __riscv_vle32_v_f32m1(ptr_k + 4 * NINPUTS, NINPUTS);
    vec_k_1 = __riscv_vle32_v_f32m1(ptr_k + 5 * NINPUTS, NINPUTS);
    vec_k_2 = __riscv_vle32_v_f32m1(ptr_k + 6 * NINPUTS, NINPUTS);
    vec_k_3 = __riscv_vle32_v_f32m1(ptr_k + 7 * NINPUTS, NINPUTS);

    vec_s_0 = __riscv_vfmacc_vf_f32m1(vec_s_0, ptr_b[4], vec_k_0, NINPUTS);
    vec_s_1 = __riscv_vfmacc_vf_f32m1(vec_s_1, ptr_b[5], vec_k_1, NINPUTS);
    vec_s_2 = __riscv_vfmacc_vf_f32m1(vec_s_2, ptr_b[6], vec_k_2, NINPUTS);
    vec_s_3 = __riscv_vfmacc_vf_f32m1(vec_s_3, ptr_b[7], vec_k_3, NINPUTS);

    // 8 - 11
    vec_k_0 = __riscv_vle32_v_f32m1(ptr_k +  8 * NINPUTS, NINPUTS);
    vec_k_1 = __riscv_vle32_v_f32m1(ptr_k +  9 * NINPUTS, NINPUTS);
    vec_k_2 = __riscv_vle32_v_f32m1(ptr_k + 10 * NINPUTS, NINPUTS);
    vec_k_3 = __riscv_vle32_v_f32m1(ptr_k + 11 * NINPUTS, NINPUTS);

    vec_s_0 = __riscv_vfmacc_vf_f32m1(vec_s_0, ptr_b[ 8], vec_k_0, NINPUTS);
    vec_s_1 = __riscv_vfmacc_vf_f32m1(vec_s_1, ptr_b[ 9], vec_k_1, NINPUTS);
    vec_s_2 = __riscv_vfmacc_vf_f32m1(vec_s_2, ptr_b[10], vec_k_2, NINPUTS);
    vec_s_3 = __riscv_vfmacc_vf_f32m1(vec_s_3, ptr_b[11], vec_k_3, NINPUTS);

    // 12 - 15
    vec_k_0 = __riscv_vle32_v_f32m1(ptr_k + 12 * NINPUTS, NINPUTS);
    vec_k_1 = __riscv_vle32_v_f32m1(ptr_k + 13 * NINPUTS, NINPUTS);
    vec_k_2 = __riscv_vle32_v_f32m1(ptr_k + 14 * NINPUTS, NINPUTS);
    vec_k_3 = __riscv_vle32_v_f32m1(ptr_k + 15 * NINPUTS, NINPUTS);

    vec_s_0 = __riscv_vfmacc_vf_f32m1(vec_s_0, ptr_b[12], vec_k_0, NINPUTS);
    vec_s_1 = __riscv_vfmacc_vf_f32m1(vec_s_1, ptr_b[13], vec_k_1, NINPUTS);
    vec_s_2 = __riscv_vfmacc_vf_f32m1(vec_s_2, ptr_b[14], vec_k_2, NINPUTS);
    vec_s_3 = __riscv_vfmacc_vf_f32m1(vec_s_3, ptr_b[15], vec_k_3, NINPUTS);

    vec_s_0 = __riscv_vfadd_vv_f32m1(vec_s_0, vec_s_1, NINPUTS);
    vec_s_2 = __riscv_vfadd_vv_f32m1(vec_s_2, vec_s_3, NINPUTS);
    vec_s_4 = __riscv_vfadd_vv_f32m1(vec_s_0, vec_s_2, NINPUTS);

    // sum and negation
    vec_b = __riscv_vle32_v_f32m1(ptr_d, 4);
    vec_s_0 = __riscv_vfadd_vv_f32m1(vec_s_4, vec_b, 4);
    vec_s_1 = __riscv_vfneg_v_f32m1(vec_s_0, 4);
    __riscv_vse32_v_f32m1(ptr_u, vec_s_1, 4);
}


// // // x[:, i+1] = Adyn * x[:, i] + Bdyn * u[:, i]
// inline void forward_pass_2(TinySolver *solver, int i) {
// #ifdef USE_MATVEC
//     matvec(solver->work->Adyn.data, solver->work->x.vector[i], solver->work->x1.data, NSTATES, NSTATES);
//     matvec(solver->work->Bdyn.data, solver->work->u.vector[i], solver->work->x2.data, NSTATES, NINPUTS);
// #else
//     matmul(solver->work->x.vector[i], solver->work->Adyn.data, solver->work->x1.data, 1, NSTATES, NSTATES);
//     matmul(solver->work->u.vector[i], solver->work->Bdyn.data, solver->work->x2.data, 1, NSTATES, NINPUTS);
// #endif
//     matadd(solver->work->x1.data, solver->work->x2.data, solver->work->x.vector[i + 1], 1, NSTATES);
//     TRACE_CHECKSUM(forward_pass_2, solver->work->x);
// }


inline void forward_pass_2(TinySolver *solver, int i) {
    vfloat32m1_t vec_a_0, vec_a_1, vec_a_2, vec_a_3;
    vfloat32m1_t vec_b_0, vec_b_1, vec_b_2, vec_b_3;
    vfloat32m1_t vec_s_0, vec_s_1, vec_s_2, vec_s_3, vec_s_4;
    vfloat32m1_t vec_sum_0, vec_sum_1, vec_sum_2, vec_sum_3;
    vfloat32m1_t vec_zero = __riscv_vfsub_vv_f32m1(vec_zero, vec_zero, NSTATES);

    float *ptr_a = solver->work->AdynT.data;
    float *ptr_b = solver->work->BdynT.data;
    float *ptr_x = solver->work->x.vector[i];
    float *ptr_u = solver->work->u.vector[i];
    float *ptr_x_new = solver->work->x.vector[i+1];

    vec_s_0 = vec_zero;
    for (int j = 0; j < NSTATES; j++) {
        vec_a_0 = __riscv_vle32_v_f32m1(ptr_a + j * NSTATES, NSTATES);
        vec_s_0 = __riscv_vfmacc_vf_f32m1(vec_s_0, ptr_x[j], vec_a_0, NSTATES);
    }
    // __riscv_vse32_v_f32(ptr_x1, vec_s_0, NSTATES);

    vec_s_4 = vec_zero;
    for (int j = 0; j < NINPUTS; j++) {
        vec_b_0 = __riscv_vle32_v_f32m1(ptr_b + j * NSTATES, NSTATES);
        vec_s_4 = __riscv_vfmacc_vf_f32m1(vec_s_4, ptr_u[j], vec_b_0, NSTATES);
    }
    // __riscv_vse32_v_f32(ptr_x2, vec_s_0, NSTATES);
    vec_s_0 = __riscv_vfadd_vv_f32m1(vec_s_0, vec_s_4, NSTATES);
    __riscv_vse32_v_f32(ptr_x_new, vec_s_0, NSTATES);

    // matvec(solver->work->Adyn.data, solver->work->x.col(i), solver->work->x1.data, NSTATES, NSTATES);
    // matvec(solver->work->Bdyn.data, solver->work->u.col(i), solver->work->x2.data, NSTATES, NINPUTS);
    // matadd(solver->work->x1.data, solver->work->x2.data, solver->work->x.col(i + 1), 1, NSTATES);
}


// // d[:, i] = Quu_inv * (BdynT * p[:, i+1] + r[:, i]);
// inline void backward_pass_1(TinySolver *solver, int i) {
// #ifdef USE_MATVEC
//     matvec(solver->work->BdynT.data, solver->work->p.vector[i + 1], solver->work->u1.data, NINPUTS, NSTATES);
//     matadd(solver->work->r.vector[i], solver->work->u1.data, solver->work->u2.data, 1, NINPUTS);
//     matvec(solver->cache->Quu_inv.data, solver->work->u2.data, solver->work->d.vector[i], NINPUTS, NINPUTS);
// #else
//     matmul(solver->work->p.vector[i + 1], solver->work->BdynT.data, solver->work->u1.data, 1, NINPUTS, NSTATES);
//     matadd(solver->work->r.vector[i], solver->work->u1.data, solver->work->u2.data, 1, NINPUTS);
//     matmul(solver->work->u2.data, solver->cache->Quu_inv.data, solver->work->d.vector[i], 1, NINPUTS, NINPUTS);
// #endif
//     TRACE_CHECKSUM(backward_pass_1, solver->work->d);
// }

inline void backward_pass_1(TinySolver *solver, int i) {

    // vfloat32m1_t vec_b_0, vec_b_1, vec_b_2, vec_a_3;
    vfloat32m1_t vec_a_0, vec_a_1, vec_a_2, vec_a_3;
    vfloat32m1_t vec_q_0, vec_q_1, vec_q_2, vec_q_3;
    vfloat32m1_t vec_b_0, vec_b_1, vec_b_2, vec_b_3;
    vfloat32m1_t vec_r;
    vfloat32m1_t vec_s_0, vec_s_1, vec_s_2, vec_s_3, vec_s_4;
    vfloat32m1_t vec_sum_0, vec_sum_1, vec_sum_2, vec_sum_3;
    vfloat32m1_t vec_zero = __riscv_vfsub_vv_f32m1(vec_zero, vec_zero, NSTATES);

    float *ptr_b = solver->work->Bdyn.data;
    float *ptr_q = solver->cache->Quu_inv.data;
    float *ptr_p = solver->work->p.vector[i+1];
    float *ptr_u = solver->work->u.vector[i];
    float *ptr_u1 = solver->work->u1.vector[0];

    float *ptr_r = solver->work->r.vector[i];
    float *ptr_d = solver->work->d.vector[i];


    vec_s_0 = vec_zero;
    for(int j = 0; j < NSTATES; j++) {
        vec_b_0 = __riscv_vle32_v_f32m1(ptr_b + j * NINPUTS, NINPUTS);
        vec_s_0 = __riscv_vfmacc_vf_f32m1(vec_s_0, ptr_p[j], vec_b_0, NINPUTS);
    }

    vec_r = __riscv_vle32_v_f32m1(ptr_r, NINPUTS);
    vec_s_4 = __riscv_vfadd_vv_f32m1(vec_s_0, vec_r, NINPUTS);

    vec_s_0 = vec_zero;
    for(int j = 0; j < NINPUTS; j++) {
        vec_q_0 = __riscv_vle32_v_f32m1(ptr_q + j * NINPUTS, NINPUTS);
        vec_s_0 = __riscv_vfmul_vv_f32m1(vec_q_0, vec_s_4, 12);
        vec_sum_0 = __riscv_vfredusum_vs_f32m1_f32m1(vec_s_0, vec_zero, NINPUTS);
        ptr_d[j] = __riscv_vfmv_f_s_f32m1_f32(vec_sum_0);
    }

    
    // matvec(solver->work->BdynT.data, solver->work->p.col(i + 1), solver->work->u1.data, NINPUTS, NSTATES);
    // matadd(solver->work->r.col(i), solver->work->u1.data, solver->work->u2.data, 1, NINPUTS);
    // matvec(solver->cache->Quu_inv.data, solver->work->u2.data, solver->work->d.col(i), NINPUTS, NINPUTS);
}



// // p[:, i] = q[:, i] + AmBKt * p[:, i + 1] - KinfT * r[:, i]
// inline void backward_pass_2(TinySolver *solver, int i) {
// #ifdef USE_MATVEC
//     matvec(solver->cache->AmBKt.data, solver->work->p.vector[i + 1], solver->work->x1.data, NSTATES, NSTATES);
//     matvec(solver->cache->KinfT.data, solver->work->r.vector[i], solver->work->x2.data, NSTATES, NINPUTS);
// #else
//     matmul(solver->work->p.vector[i + 1], solver->cache->AmBKt.data, solver->work->x1.data, 1, NSTATES, NSTATES);
//     matmul(solver->work->r.vector[i], solver->cache->KinfT.data, solver->work->x2.data, 1, NSTATES, NINPUTS);
// #endif
//     matsub(solver->work->x1.data, solver->work->x2.data, solver->work->x3.data, 1, NSTATES);
//     matadd(solver->work->x3.data, solver->work->q.vector[i], solver->work->p.vector[i], 1, NSTATES);
//     TRACE_CHECKSUM(backward_pass_2, solver->work->p);
// }

// TODO FIX AMBKt
inline void backward_pass_2(TinySolver *solver, int i) {

    vfloat32m1_t vec_a_0, vec_a_1, vec_a_2, vec_a_3;
    vfloat32m1_t vec_k_0, vec_k_1, vec_k_2, vec_k_3;
    vfloat32m1_t vec_b_0, vec_b_1, vec_b_2, vec_b_3;
    vfloat32m1_t vec_s_0, vec_s_1, vec_s_2, vec_s_3, vec_s_4;
    vfloat32m1_t vec_sum_0, vec_sum_1, vec_sum_2, vec_sum_3;
    vfloat32m1_t vec_zero = __riscv_vfsub_vv_f32m1(vec_zero, vec_zero, NSTATES);

    // TODO FIX
    float *ptr_a = solver->cache->AmBKtT.data;
    // float *ptr_a = solver->cache->AmBKt.data;
    float *ptr_k = solver->cache->Kinf.data;
    float *ptr_b = solver->work->BdynT.data;
    float *ptr_p = solver->work->p.vector[i+1];
    float *ptr_p_new = solver->work->p.vector[i];
    float *ptr_r = solver->work->r.vector[i];
    float *ptr_q = solver->work->q.vector[i];

    vec_s_0 = vec_zero;
    for (int j = 0; j < NSTATES; j++) {
        vec_a_0 = __riscv_vle32_v_f32m1(ptr_a + j * NSTATES, NSTATES);
        vec_s_0 = __riscv_vfmacc_vf_f32m1(vec_s_0, ptr_p[j], vec_a_0, NSTATES);
    }

    vec_s_4 = vec_zero;
    for (int j = 0; j < NINPUTS; j++) {
        vec_k_0 = __riscv_vle32_v_f32m1(ptr_k + j * NSTATES, NSTATES);
        vec_s_4 = __riscv_vfmacc_vf_f32m1(vec_s_4, ptr_r[j], vec_k_0, NSTATES);
    }

    vec_s_0 = __riscv_vfsub_vv_f32m1(vec_s_0, vec_s_4, NSTATES);
    vec_s_1 = __riscv_vle32_v_f32m1(ptr_q, NSTATES);
    vec_s_0 = __riscv_vfadd_vv_f32m1(vec_s_0, vec_s_1, NSTATES);
    __riscv_vse32_v_f32m1(ptr_p_new, vec_s_0, NSTATES);

    // matvec(solver->cache->AmBKt.data, solver->work->p.col(i + 1), solver->work->x1.data, NSTATES, NSTATES);
    // matvec(solver->cache->KinfT.data, solver->work->r.col(i), solver->work->x2.data, NSTATES, NINPUTS);
    // matsub(solver->work->x1.data, solver->work->x2.data, solver->work->x3.data, 1, NSTATES);
    // matadd(solver->work->x3.data, solver->work->q.col(i), solver->work->p.col(i), 1, NSTATES);
}

// // y u znew  g x vnew
// inline void update_dual_1(TinySolver *solver) {
//     matadd(solver->work->y.data, solver->work->u.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
//     matsub(solver->work->m1.data, solver->work->znew.data, solver->work->y.data, NHORIZON - 1, NINPUTS);
//     matadd(solver->work->g.data, solver->work->x.data, solver->work->s1.data, NHORIZON, NSTATES);
//     matsub(solver->work->s1.data, solver->work->vnew.data, solver->work->g.data, NHORIZON, NSTATES);
//     TRACE_CHECKSUM(update_dual_1, solver->work->g);
// }

inline void update_dual_1(TinySolver *solver) {

    int k = (NHORIZON -1 )* NINPUTS;
    int j = (NHORIZON)* NSTATES;

    vfloat32m1_t vec_y, vec_u, vec_znew;
    vfloat32m1_t vec_g, vec_x, vec_vnew;
    vfloat32m1_t vec_s;

    float *ptr_y = solver->work->y.vector[0];
    float *ptr_u = solver->work->u.vector[0];
    float *ptr_znew = solver->work->znew.vector[0];

    float *ptr_g = solver->work->g.vector[0];
    float *ptr_x = solver->work->x.vector[0];
    float *ptr_vnew = solver->work->vnew.vector[0];

    for (size_t vl; k > 0; k -= vl, ptr_y += vl, ptr_u += vl, ptr_znew += vl) {
        vl = __riscv_vsetvl_e32(k);
        vec_y = __riscv_vle32_v_f32(ptr_y, vl);
        vec_u = __riscv_vle32_v_f32(ptr_u, vl);
        vec_znew = __riscv_vle32_v_f32(ptr_znew, vl);
        vec_s = __riscv_vfadd_vv_f32(vec_y, vec_u, vl);
        vec_s = __riscv_vfsub_vv_f32(vec_s, vec_znew, vl);
        __riscv_vse32_v_f32(ptr_y, vec_s, vl);
    }

    for (size_t vl; j > 0; j -= vl, ptr_y += vl, ptr_u += vl, ptr_znew += vl) {
        vl = __riscv_vsetvl_e32(j);
        vec_g = __riscv_vle32_v_f32(ptr_g, vl);
        vec_x = __riscv_vle32_v_f32(ptr_x, vl);
        vec_vnew = __riscv_vle32_v_f32(ptr_vnew, vl);
        vec_s = __riscv_vfadd_vv_f32(vec_g, vec_x, vl);
        vec_s = __riscv_vfsub_vv_f32(vec_s, vec_vnew, vl);
        __riscv_vse32_v_f32(ptr_g, vec_s, vl);
    }

    // matadd(solver->work->y.data, solver->work->u.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    // matsub(solver->work->m1.data, solver->work->znew.data, solver->work->y.data, NHORIZON - 1, NINPUTS);
    // matadd(solver->work->g.data, solver->work->x.data, solver->work->s1.data, NHORIZON, NSTATES);
    // matsub(solver->work->s1.data, solver->work->vnew.data, solver->work->g.data, NHORIZON, NSTATES);
}

// // Box constraints on input
// inline void update_slack_1(TinySolver *solver) {
//     matadd(solver->work->u.data, solver->work->y.data, solver->work->znew.data, NHORIZON - 1, NINPUTS);
//     if (solver->settings->en_input_bound) {
//         cwisemax(solver->work->u_min.data, solver->work->znew.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
//         cwisemin(solver->work->u_max.data, solver->work->m1.data, solver->work->znew.data, NHORIZON - 1, NINPUTS);
//     }
//     TRACE_CHECKSUM(update_slack_1, solver->work->znew);
// }

inline void update_slack_1(TinySolver *solver) {
    int k = (NHORIZON -1 )* NINPUTS;

    vfloat32m1_t vec_y, vec_u, vec_znew;
    vfloat32m1_t vec_u_min, vec_u_max;
    vfloat32m1_t vec_s;

    float *ptr_y = solver->work->y.vector[0];
    float *ptr_u = solver->work->u.vector[0];
    float *ptr_u_min = solver->work->u_min.vector[0];
    float *ptr_u_max = solver->work->u_max.vector[0];
    float *ptr_znew = solver->work->znew.vector[0];


    if (solver->settings->en_input_bound) {
        for (size_t vl; k > 0; k -= vl, ptr_y += vl, ptr_u += vl, ptr_u_min += vl, ptr_u_max += vl, ptr_znew += vl) {
            vl = __riscv_vsetvl_e32(k);
            vec_y = __riscv_vle32_v_f32(ptr_y, vl);
            vec_u = __riscv_vle32_v_f32(ptr_u, vl);
            vec_u_min = __riscv_vle32_v_f32(ptr_u_min, vl);
            vec_u_max = __riscv_vle32_v_f32(ptr_u_max, vl);
            vec_s = __riscv_vfadd_vv_f32(vec_u, vec_y, vl);
            vec_s = __riscv_vfmax_vv_f32(vec_u_min, vec_s, vl);
            vec_s = __riscv_vfmin_vv_f32(vec_u_max, vec_s, vl);
            __riscv_vse32_v_f32(ptr_znew, vec_s, vl);
        }
    } else {
        matadd(solver->work->u.data, solver->work->y.data, solver->work->znew.data, NHORIZON - 1, NINPUTS);
    }
}

// // Box constraints on state
// inline void update_slack_2(TinySolver *solver) {
//     matadd(solver->work->x.data, solver->work->g.data, solver->work->vnew.data, NHORIZON, NSTATES);
//     if (solver->settings->en_state_bound) {
//         cwisemax(solver->work->x_min.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
//         cwisemin(solver->work->x_max.data, solver->work->s1.data, solver->work->vnew.data, NHORIZON, NSTATES);
//     }
//     TRACE_CHECKSUM(update_slack_2, solver->work->vnew);
// }

inline void update_slack_2(TinySolver *solver) {
    int k = NHORIZON * NSTATES;

    vfloat32m1_t vec_x, vec_g, vec_vnew;
    vfloat32m1_t vec_x_min, vec_x_max;
    vfloat32m1_t vec_s;

    float *ptr_x = solver->work->x.vector[0];
    float *ptr_g = solver->work->g.vector[0];
    float *ptr_vnew = solver->work->vnew.vector[0];
    float *ptr_x_min = solver->work->x_min.vector[0];
    float *ptr_x_max = solver->work->x_max.vector[0];

    if (solver->settings->en_state_bound) {
        for (size_t vl; k > 0; k -= vl, ptr_x += vl, ptr_g += vl, ptr_x_min += vl, ptr_x_max += vl, ptr_vnew += vl) {
            vl = __riscv_vsetvl_e32(k);
            vec_x = __riscv_vle32_v_f32(ptr_x, vl);
            vec_g = __riscv_vle32_v_f32(ptr_g, vl);
            vec_x_min = __riscv_vle32_v_f32(ptr_x_min, vl);
            vec_x_max = __riscv_vle32_v_f32(ptr_x_max, vl);

            vec_s = __riscv_vfadd_vv_f32(vec_x, vec_g, vl); // Add x and g
            vec_s = __riscv_vfmax_vv_f32(vec_x_min, vec_s, vl); // Max of x_min and result
            vec_s = __riscv_vfmin_vv_f32(vec_x_max, vec_s, vl); // Min of x_max and result
            __riscv_vse32_v_f32(ptr_vnew, vec_s, vl); // Store result back in vnew
        }
    } else {
        for (size_t vl; k > 0; k -= vl, ptr_x += vl, ptr_g += vl, ptr_vnew += vl) {
            vl = __riscv_vsetvl_e32(k);
            vec_x = __riscv_vle32_v_f32(ptr_x, vl);
            vec_g = __riscv_vle32_v_f32(ptr_g, vl);
            vec_vnew = __riscv_vfadd_vv_f32(vec_x, vec_g, vl);
            __riscv_vse32_v_f32(ptr_vnew, vec_vnew, vl);
        }
    }
}

// inline void primal_residual_state(TinySolver *solver) {
//     matsub(solver->work->x.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
//     cwiseabs(solver->work->s1.data, solver->work->s2.data, NHORIZON, NSTATES);
//     solver->work->primal_residual_state = maxcoeff(solver->work->s2.data, NHORIZON, NSTATES);
//     TRACE_CHECKSUM(primal_residual_state, solver->work->s2);
// }

inline void primal_residual_state(TinySolver *solver) {
    int k = NHORIZON * NSTATES;

    vfloat32m1_t vec_x, vec_vnew, vec_s;
    vfloat32m1_t vec_max = __riscv_vfmv_v_f_f32m1(0.0, 16);  // Initialize vec_max with the smallest float

    float *ptr_x = solver->work->x.vector[0];
    float *ptr_vnew = solver->work->vnew.vector[0];

    for (size_t vl; k > 0; k -= vl, ptr_x += vl, ptr_vnew += vl) {
        vl = __riscv_vsetvl_e32m1(k);  // Set the vector length for this iteration
        vec_x = __riscv_vle32_v_f32m1(ptr_x, vl);  // Load elements into vector vec_v
        vec_vnew = __riscv_vle32_v_f32m1(ptr_vnew, vl);  // Load elements into vector vec_vnew
        vec_s = __riscv_vfsub_vv_f32m1(vec_x, vec_vnew, vl);  // Element-wise subtraction
        vec_s = __riscv_vfabs_v_f32m1(vec_s, vl);  // Absolute value of the difference
        vec_max = __riscv_vfmax_vv_f32m1(vec_s, vec_max, vl);  // Compute the max of current and previous values
    }

    float max_residual = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(vec_max, vec_max, 16));  // Reduction to get the maximum value
    solver->work->primal_residual_state = max_residual * solver->cache->rho;  // Scale by penalty term and store result
}

// inline void dual_residual_state(TinySolver *solver) {
//     matsub(solver->work->v.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
//     cwiseabs(solver->work->s1.data, solver->work->s2.data, NHORIZON, NSTATES);
//     solver->work->dual_residual_state = maxcoeff(solver->work->s2.data, NHORIZON, NSTATES) * solver->cache->rho;
//     TRACE_CHECKSUM(dual_residual_state, solver->work->s2);
// }

inline void dual_residual_state(TinySolver *solver) {
    int k = (NHORIZON) * NSTATES;

    vfloat32m1_t vec_v, vec_vnew, vec_s;
    vfloat32m1_t vec_max = __riscv_vfmv_v_f_f32m1(0.0, 16);  // Initialize vec_max with the smallest float

    float *ptr_v = solver->work->v.vector[0];
    float *ptr_vnew = solver->work->vnew.vector[0];

    for (size_t vl; k > 0; k -= vl, ptr_v += vl, ptr_vnew += vl) {
        vl = __riscv_vsetvl_e32m1(k);  // Set the vector length for this iteration
        vec_v = __riscv_vle32_v_f32m1(ptr_v, vl);  // Load elements into vector vec_v
        vec_vnew = __riscv_vle32_v_f32m1(ptr_vnew, vl);  // Load elements into vector vec_vnew
        vec_s = __riscv_vfsub_vv_f32m1(vec_v, vec_vnew, vl);  // Element-wise subtraction
        vec_s = __riscv_vfabs_v_f32m1(vec_s, vl);  // Absolute value of the difference
        vec_max = __riscv_vfmax_vv_f32m1(vec_s, vec_max, vl);  // Compute the max of current and previous values
    }

    float max_residual = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(vec_max, vec_max, 16));  // Reduction to get the maximum value
    solver->work->dual_residual_state = max_residual * solver->cache->rho;  // Scale by penalty term and store result

    // matsub(solver->work->v.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
    // cwiseabs(solver->work->s1.data, solver->work->s2.data, NHORIZON, NSTATES);
    // solver->work->dual_residual_state = maxcoeff(solver->work->s2.data, NHORIZON, NSTATES) * solver->cache->rho;
}

// inline void primal_residual_input(TinySolver *solver) {
//     matsub(solver->work->u.data, solver->work->znew.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
//     cwiseabs(solver->work->m1.data, solver->work->m2.data, NHORIZON - 1, NINPUTS);
//     solver->work->primal_residual_input = maxcoeff(solver->work->m2.data, NHORIZON - 1, NINPUTS);
//     TRACE_CHECKSUM(primal_residual_input, solver->work->m2);
// }

inline void primal_residual_input(TinySolver *solver) {
    int k = (NHORIZON - 1) * NINPUTS;

    vfloat32m1_t vec_u, vec_znew, vec_s;
    vfloat32m1_t vec_max = __riscv_vfmv_v_f_f32m1(0, 16);

    float *ptr_u = solver->work->u.vector[0];
    float *ptr_znew = solver->work->znew.vector[0];

    for (size_t vl; k > 0; k -= vl, ptr_u += vl, ptr_znew += vl) {
        vl = __riscv_vsetvl_e32m1(k);
        vec_u = __riscv_vle32_v_f32m1(ptr_u, vl);
        vec_znew = __riscv_vle32_v_f32m1(ptr_znew, vl);
        vec_s = __riscv_vfsub_vv_f32m1(vec_u, vec_znew, vl);
        vec_s = __riscv_vfabs_v_f32m1(vec_s, vl);
        vec_max = __riscv_vfmax_vv_f32m1(vec_s, vec_max, vl);
    }

    float max_residual = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(vec_max, vec_max, 16));
    solver->work->primal_residual_input = max_residual;
}

// inline void dual_residual_input(TinySolver *solver) {
//     matsub(solver->work->z.data, solver->work->znew.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
//     cwiseabs(solver->work->m1.data, solver->work->m2.data, NHORIZON - 1, NINPUTS);
//     solver->work->dual_residual_input = maxcoeff(solver->work->m2.data, NHORIZON - 1, NINPUTS) * solver->cache->rho;
//     TRACE_CHECKSUM(dual_residual_input, solver->work->m2);
// }

inline void dual_residual_input(TinySolver *solver) {
    int k = (NHORIZON - 1) * NINPUTS;

    vfloat32m1_t vec_z, vec_znew, vec_s;
    vfloat32m1_t vec_max = __riscv_vfmv_v_f_f32m1(0.0, 16);

    float *ptr_z = solver->work->z.vector[0];
    float *ptr_znew = solver->work->znew.vector[0];

    for (size_t vl; k > 0; k -= vl, ptr_z += vl, ptr_znew += vl) {
        vl = __riscv_vsetvl_e32m1(k);
        vec_z = __riscv_vle32_v_f32m1(ptr_z, vl);
        vec_znew = __riscv_vle32_v_f32m1(ptr_znew, vl);
        vec_s = __riscv_vfsub_vv_f32m1(vec_z, vec_znew, vl);
        vec_s = __riscv_vfabs_v_f32m1(vec_s, vl);
        vec_max = __riscv_vfmax_vv_f32m1(vec_s, vec_max, vl);
    }

    float max_residual = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(vec_max, vec_max, 16)) * solver->cache->rho;
    solver->work->dual_residual_input = max_residual;
}

// inline void update_linear_cost_1(TinySolver *solver) {
//     matsub(solver->work->znew.data, solver->work->y.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
//     matmulf(solver->work->m1.data, solver->work->r.data, -solver->cache->rho, NHORIZON - 1, NINPUTS);
//     TRACE_CHECKSUM(update_linear_cost_1, solver->work->r);
// }

inline void update_linear_cost_1(TinySolver *solver) {
    int k = (NHORIZON -1 )* NINPUTS;

    vfloat32m1_t vec_y, vec_u, vec_znew;
    vfloat32m1_t vec_g, vec_x, vec_vnew;
    vfloat32m1_t vec_s;

    float *ptr_y = solver->work->y.vector[0];
    float *ptr_r = solver->work->r.vector[0];
    float *ptr_znew = solver->work->znew.vector[0];
    
    const float rho = -solver->cache->rho;


    for (size_t vl; k > 0; k -= vl, ptr_y += vl, ptr_r += vl, ptr_znew += vl) {
        vl = __riscv_vsetvl_e32(k);
        vec_y = __riscv_vle32_v_f32(ptr_y, vl);
        vec_znew = __riscv_vle32_v_f32(ptr_znew, vl);
        vec_s = __riscv_vfsub_vv_f32(vec_znew, vec_y, vl);
        vec_s = __riscv_vfmul_vf_f32(vec_s, rho, vl);
        __riscv_vse32_v_f32(ptr_r, vec_s, vl);
    }


    // matsub(solver->work->znew.data, solver->work->y.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    // matmulf(solver->work->m1.data, solver->work->r.data, -solver->cache->rho, NHORIZON - 1, NINPUTS);
}

// inline void update_linear_cost_2(TinySolver *solver, int i) {
//     cwisemul(solver->work->Xref.vector[i], solver->work->Q.data, solver->work->x1.data, 1, NSTATES);
//     matneg(solver->work->x1.data, solver->work->q.vector[i], 1, NSTATES);
//     TRACE_CHECKSUM(update_linear_cost_2, solver->work->q);
// }

inline void update_linear_cost_2(TinySolver *solver, int i) {
    int k = NSTATES;  // Assuming NSTATES is the number of elements in the column

    vfloat32m1_t vec_Xref, vec_Q, vec_s;
    
    float *ptr_Xref = solver->work->Xref.vector[i];  // Assuming col(i) returns a pointer to the column
    float *ptr_Q = solver->work->Q.data;  // Assuming Q is a matrix stored in row-major order
    float *ptr_q = solver->work->q.vector[i];  // Assuming q is organized similarly to Xref

    for (size_t vl; k > 0; k -= vl, ptr_Xref += vl, ptr_Q += vl) {
        vl = __riscv_vsetvl_e32m1(k);
        vec_Xref = __riscv_vle32_v_f32m1(ptr_Xref, vl);
        vec_Q = __riscv_vle32_v_f32m1(ptr_Q, vl);
        vec_s = __riscv_vfmul_vv_f32m1(vec_Xref, vec_Q, vl);  // Element-wise multiplication
        vec_s = __riscv_vfneg_v_f32m1(vec_s, vl);  // Negate the result
        __riscv_vse32_v_f32m1(ptr_q, vec_s, vl);  // Store the result back in q
    }
}

// inline void update_linear_cost_3(TinySolver *solver) {
//     matsub(solver->work->vnew.data, solver->work->g.data, solver->work->s1.data, NHORIZON, NSTATES);
//     matmulf(solver->work->s1.data, solver->work->s2.data, solver->cache->rho, NHORIZON, NSTATES);
//     matsub(solver->work->q.data, solver->work->s2.data, solver->work->s1.data, NHORIZON, NSTATES);
//     matsetv(solver->work->q.data, solver->work->s1.data, solver->work->q.outer, solver->work->q.inner);
//     TRACE_CHECKSUM(update_linear_cost_3, solver->work->s1);
// }

inline void update_linear_cost_3(TinySolver *solver) {
    int k = (NHORIZON)* NSTATES;

    vfloat32m1_t vec_g, vec_q, vec_vnew;
    vfloat32m1_t vec_s;
    float *ptr_g = solver->work->g.vector[0];
    float *ptr_q = solver->work->q.vector[0];
    float *ptr_vnew = solver->work->vnew.vector[0];
    
    const float rho = solver->cache->rho;


    for (size_t vl; k > 0; k -= vl, ptr_g += vl, ptr_q += vl, ptr_vnew += vl) {
        vl = __riscv_vsetvl_e32(k);
        vec_g = __riscv_vle32_v_f32(ptr_g, vl);
        vec_q = __riscv_vle32_v_f32(ptr_q, vl);
        vec_vnew = __riscv_vle32_v_f32(ptr_vnew, vl);
        vec_s = __riscv_vfsub_vv_f32(vec_vnew, vec_g, vl);
        vec_s = __riscv_vfmul_vf_f32(vec_s, rho, vl);
        vec_s = __riscv_vfsub_vv_f32(vec_q, vec_s, vl);
        // TODO why does this not work?
        // vec_s = __riscv_vfmsac_vf_f32(vec_q, rho, vec_s, vl);
        __riscv_vse32_v_f32(ptr_q, vec_s, vl);
    }

    // matsub(solver->work->vnew.data, solver->work->g.data, solver->work->s1.data, NHORIZON, NSTATES);
    // matmulf(solver->work->s1.data, solver->work->s2.data, solver->cache->rho, NHORIZON, NSTATES);
    // matsub(solver->work->q.data, solver->work->s2.data, solver->work->s1.data, NHORIZON, NSTATES);
    // solver->work->q.set(solver->work->s1._data);
}


inline void update_linear_cost_4(TinySolver *solver) {
    matsub(solver->work->vnew.vector[NHORIZON - 1], solver->work->g.vector[NHORIZON - 1], solver->work->x1.data, 1, NSTATES);
    matmulf(solver->work->x1.data, solver->work->x2.data, solver->cache->rho, 1, NSTATES);
#ifdef USE_MATVEC
    matvec(solver->cache->PinfT.data, solver->work->Xref.vector[NHORIZON - 1], solver->work->x1.data, NSTATES, NSTATES);
#else
    matmul(solver->work->Xref.vector[NHORIZON - 1], solver->cache->PinfT.data, solver->work->x1.data, 1, NSTATES, NSTATES);
#endif
    matadd(solver->work->x1.data, solver->work->x2.data, solver->work->x3.data, 1, NSTATES);
    matneg(solver->work->x3.data, solver->work->p.vector[NHORIZON - 1], 1, NSTATES);
    TRACE_CHECKSUM(update_linear_cost_4, solver->work->p);
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


};
#endif //TINYMPC_ADMM_RVV_HPP
