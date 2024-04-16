//
// Created by widyadewi on 2/23/24.
//

#pragma once
#ifndef TINYMPC_ADMM_RVV_HPP
#define TINYMPC_ADMM_RVV_HPP

#include "types_rvv.hpp"
#include "riscv_vector.h"
#include <string.h>

extern "C" {

// vec_1 = a b c d e f g h;
// vec_idx = 1 2 3 4 1 2 3 4;

// vec_perm vd, vec_1, vec_idx
// vec_perm = a b c d a b c 

// b1 b2 b3 b4 b1 b2 b3 b4 b1 b2 b3 b4 b1 b2 b3 b4
// a11 a12 ...

// b1 b2 b3 b4 .......
// a11 a12 a13 a14 ......


// b1 b1 b1 b1
// a11 a21 a31 a41

// c1 c2 c3 c4

// for (i in len(b))
//     vec_b = broadcast(b[i])
//     vec_a = A.col(i)
//     vec_s = fmac(vec_b, vec_a, vec_s)

// store(vec_s, c);





// freddo(a) 

inline void matvec_rvv_12x4_golden(float ** a, float **b, float **c) {
    float vec_a[16];
    float vec_b[16];
    float vec_s[16];
    float vec_sum[16];

    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    float *ptr_c = &c[0][0];
    float sum;

    memcpy(vec_b, ptr_b, 12 * sizeof(float));

    memcpy(vec_a, ptr_a, 12 * sizeof(float));
    for (int i = 0; i < 12; i++) {
        vec_s[i] = vec_a[i] * vec_b[i];
    }

    sum = 0;
    for (int i = 0; i < 12; i++) {
        sum += vec_s[i];
    }
    ptr_c[0] = sum;

    memcpy(vec_a, ptr_a + 12, 12 * sizeof(float));
    for (int i = 0; i < 12; i++) {
        vec_s[i] = vec_a[i] * vec_b[i];
    }

    sum = 0;
    for (int i = 0; i < 12; i++) {
        sum += vec_s[i];
    }
    ptr_c[1] = sum;

    memcpy(vec_a, ptr_a + 24, 12 * sizeof(float));
    for (int i = 0; i < 12; i++) {
        vec_s[i] = vec_a[i] * vec_b[i];
    }

    sum = 0;
    for (int i = 0; i < 12; i++) {
        sum += vec_s[i];
    }
    ptr_c[2] = sum;

    memcpy(vec_a, ptr_a + 36, 12 * sizeof(float));
    for (int i = 0; i < 12; i++) {
        vec_s[i] = vec_a[i] * vec_b[i];
    }

    sum = 0;
    for (int i = 0; i < 12; i++) {
        sum += vec_s[i];
    }
    ptr_c[3] = sum;

}

inline void matvec_rvv_12x4(float ** a, float **b, float **c) {
    vfloat32m1_t vec_a;
    vfloat32m1_t vec_b;
    vfloat32m1_t vec_s;
    vfloat32m1_t vec_sum;
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, 12);

    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    float *ptr_c = &c[0][0];
    float sum;
    vec_b = __riscv_vle32_v_f32m1(ptr_b, 12);

    vec_a = __riscv_vle32_v_f32m1(ptr_a, 12);
    vec_s = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 12);
    vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, 12);
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    ptr_c[0] = sum;

    vec_a = __riscv_vle32_v_f32m1(ptr_a + 12, 12);
    vec_s = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 12);
    vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, 12);
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    ptr_c[1] = sum;

    vec_a = __riscv_vle32_v_f32m1(ptr_a + 24, 12);
    vec_s = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 12);
    vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, 12);
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    ptr_c[2] = sum;

    vec_a = __riscv_vle32_v_f32m1(ptr_a + 36, 12);
    vec_s = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 12);
    vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, 12);
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    ptr_c[3] = sum;
}

// inline void matvec_rvv_12x4(float ** a, float **b, float **c) {
//     vfloat32m1_t vec_a;
//     vfloat32m1_t vec_b;
//     vfloat32m1_t vec_s;
//     vfloat32m1_t vec_sum;
//     vfloat32m1_t vec_zero = __riscv_vfsub_vv_f32m1(vec_zero, vec_zero, 12); // Correct way to initialize vec_zero to all zeros

//     float *ptr_a = &a[0][0];
//     float *ptr_b = &b[0][0];
//     float *ptr_c = &c[0][0];
//     float sum;
//     vec_b = __riscv_vle32_v_f32m1(ptr_b, 12); // Load the vector b once, as it is used unchanged in all iterations

//     // Correct the loading of vec_a for each row of matrix a and subsequent computation
//     for (int i = 0; i < 4; i++) {
//         vec_a = __riscv_vle32_v_f32m1(ptr_a + i * 12, 12); // Correctly advance ptr_a for each row
//         vec_s = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 12);
//         vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, 12);
//         sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
//         ptr_c[i] = sum; // Assign the result to the correct position in c
//     }
// }

inline void forward_pass_1(TinySolver *solver, int i) {
    vfloat32m1_t vec_a;
    vfloat32m1_t vec_b;
    vfloat32m1_t vec_s;
    vfloat32m1_t vec_sum;
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, 12);

    float *ptr_a = &solver->cache->Kinf.data[0][0];
    float *ptr_b = &solver->work->x.data[i][0];
    float *ptr_d = &solver->work->d.data[i][0];
    float ptr_c[4];
    float *ptr_u = &solver->work->u.data[i][0];
    float sum;
    vec_b = __riscv_vle32_v_f32m1(ptr_b, 12);

    vec_a = __riscv_vle32_v_f32m1(ptr_a, 12);
    vec_s = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 12);
    vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, 12);
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    ptr_c[0] = sum;

    vec_s = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 12);
    vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, 12);
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    ptr_c[1] = sum;

    vec_a = __riscv_vle32_v_f32m1(ptr_a + 24, 12);
    vec_s = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 12);
    vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, 12);
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    ptr_c[2] = sum;

    vec_a = __riscv_vle32_v_f32m1(ptr_a + 36, 12);
    vec_s = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 12);
    vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, 12);
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    ptr_c[3] = sum;

    // sum and negation
    vec_a = __riscv_vle32_v_f32m1(ptr_c, 4);
    vec_b = __riscv_vle32_v_f32m1(ptr_d, 4);
    vec_s = __riscv_vfadd_vv_f32m1(vec_a, vec_b, 4);
    vec_s = __riscv_vfsub_vv_f32m1(vec_zero, vec_s, 4);
    __riscv_vse32_v_f32m1(ptr_u, vec_s, 4);
}

// u1 = x[:, i] * Kinf; u2 = u1 + d; u[:, i] = -u2
inline void forward_pass_1_old(TinySolver *solver, int i) {
    // matvec_golden(solver->cache->Kinf.data, solver->work->x.col(i), solver->work->u1.data, NINPUTS, NSTATES);
    // matvec_rvv_12x4(solver->cache->Kinf.data, solver->work->x.col(i), solver->work->u1.data);
    // matvec_rvv_12x4(solver->cache->Kinf.data, solver->work->x.col(i), solver->work->u1.data);
    matmul(solver->work->x.col(i), solver->cache->Kinf.data, solver->work->u1.data, 1, NINPUTS, NSTATES);
    matadd(solver->work->u1.data, solver->work->d.col(i), solver->work->u2.data, 1, NINPUTS);
    matneg(solver->work->u2.data, solver->work->u.col(i), 1, NINPUTS);
}

// x[:, i+1] = Adyn * x[:, i] + Bdyn * u[:, i]
inline void forward_pass_2(TinySolver *solver, int i) {
    matmul(solver->work->x.col(i), solver->work->Adyn.data, solver->work->x1.data, 1, NSTATES, NSTATES);
    matmul(solver->work->u.col(i), solver->work->Bdyn.data, solver->work->x2.data, 1, NSTATES, NINPUTS);
    matadd(solver->work->x1.data, solver->work->x2.data, solver->work->x.col(i + 1), 1, NSTATES);
}

inline void forward_pass_2_old(TinySolver *solver, int i) {
    matmul(solver->work->x.col(i), solver->work->Adyn.data, solver->work->x1.data, 1, NSTATES, NSTATES);
    matmul(solver->work->u.col(i), solver->work->Bdyn.data, solver->work->x2.data, 1, NSTATES, NINPUTS);
    matadd(solver->work->x1.data, solver->work->x2.data, solver->work->x.col(i + 1), 1, NSTATES);
}

// d[:, i] = Quu_inv * (BdynT * p[:, i+1] + r[:, i]);
inline void backward_pass_1(TinySolver *solver, int i) {
    matmul(solver->work->p.col(i + 1), solver->work->BdynT.data, solver->work->u1.data, 1, NINPUTS, NSTATES);
    matadd(solver->work->r.col(i), solver->work->u1.data, solver->work->u2.data, 1, NINPUTS);
    matmul(solver->work->u2.data, solver->cache->Quu_inv.data, solver->work->d.col(i), 1, NINPUTS, NINPUTS);
}

// p[:, i] = q[:, i] + AmBKt * p[:, i + 1] - KinfT * r[:, i]
inline void backward_pass_2(TinySolver *solver, int i) {
    matmul(solver->work->p.col(i + 1), solver->cache->AmBKt.data, solver->work->x1.data, 1, NSTATES, NSTATES);
    matmul(solver->work->r.col(i), solver->cache->KinfT.data, solver->work->x2.data, 1, NSTATES, NINPUTS);
    matsub(solver->work->x1.data, solver->work->x2.data, solver->work->x3.data, 1, NSTATES);
    matadd(solver->work->x3.data, solver->work->q.col(i), solver->work->p.col(i), 1, NSTATES);
}

// y u znew  g x vnew
inline void update_dual_1(TinySolver *solver) {
    matadd(solver->work->y.data, solver->work->u.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    matsub(solver->work->m1.data, solver->work->znew.data, solver->work->y.data, NHORIZON - 1, NINPUTS);
    matadd(solver->work->g.data, solver->work->x.data, solver->work->s1.data, NHORIZON, NSTATES);
    matsub(solver->work->s1.data, solver->work->vnew.data, solver->work->g.data, NHORIZON, NSTATES);
}

// Box constraints on input
inline void update_slack_1(TinySolver *solver) {
    matadd(solver->work->u.data, solver->work->y.data, solver->work->znew.data, NHORIZON - 1, NINPUTS);
    if (solver->settings->en_input_bound) {
        cwisemax(solver->work->u_min.data, solver->work->znew.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
        cwisemin(solver->work->u_max.data, solver->work->m1.data, solver->work->znew.data, NHORIZON - 1, NINPUTS);
    }
}

// Box constraints on state
inline void update_slack_2(TinySolver *solver) {
    matadd(solver->work->x.data, solver->work->g.data, solver->work->vnew.data, NHORIZON, NSTATES);
    if (solver->settings->en_state_bound) {
        cwisemax(solver->work->x_min.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
        cwisemin(solver->work->x_max.data, solver->work->s1.data, solver->work->vnew.data, NHORIZON, NSTATES);
    }
}

inline void primal_residual_state(TinySolver *solver) {
    matsub(solver->work->x.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
    cwiseabs(solver->work->s1.data, solver->work->s2.data, NHORIZON, NSTATES);
    solver->work->primal_residual_state = maxcoeff(solver->work->s2.data, NHORIZON, NSTATES);
}

inline void dual_residual_state(TinySolver *solver) {
    matsub(solver->work->v.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
    cwiseabs(solver->work->s1.data, solver->work->s2.data, NHORIZON, NSTATES);
    solver->work->dual_residual_state = maxcoeff(solver->work->s2.data, NHORIZON, NSTATES) * solver->cache->rho;
}

inline void primal_residual_input(TinySolver *solver) {
    matsub(solver->work->u.data, solver->work->znew.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    cwiseabs(solver->work->m1.data, solver->work->m2.data, NHORIZON - 1, NINPUTS);
    solver->work->primal_residual_input = maxcoeff(solver->work->m2.data, NHORIZON - 1, NINPUTS);
}

inline void dual_residual_input(TinySolver *solver) {
    matsub(solver->work->z.data, solver->work->znew.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    cwiseabs(solver->work->m1.data, solver->work->m2.data, NHORIZON - 1, NINPUTS);
    solver->work->dual_residual_input = maxcoeff(solver->work->m2.data, NHORIZON - 1, NINPUTS) * solver->cache->rho;
}

inline void update_linear_cost_1(TinySolver *solver) {
    matsub(solver->work->znew.data, solver->work->y.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    matmulf(solver->work->m1.data, solver->work->r.data, -solver->cache->rho, NHORIZON - 1, NINPUTS);
}

inline void update_linear_cost_2(TinySolver *solver, int i) {
    cwisemul(solver->work->Xref.col(i), solver->work->Q.data, solver->work->x1.data, 1, NSTATES);
    // solver->work->Xref.print("float", "lc2 Xref");
    // solver->work->Q.print("float", "lc2 Q");
    // solver->work->x1.print("float", "lc2 x1");
    matneg(solver->work->x1.data, solver->work->q.col(i), 1, NSTATES);
    // solver->work->q.print("float", "lc2 q");
}

inline void update_linear_cost_3(TinySolver *solver) {
    matsub(solver->work->vnew.data, solver->work->g.data, solver->work->s1.data, NHORIZON, NSTATES);
    matmulf(solver->work->s1.data, solver->work->s2.data, solver->cache->rho, NHORIZON, NSTATES);
    matsub(solver->work->q.data, solver->work->s2.data, solver->work->s1.data, NHORIZON, NSTATES);
    solver->work->q.set(solver->work->s1._data);
}

inline void update_linear_cost_4(TinySolver *solver) {
    matsub(solver->work->vnew.col(NHORIZON - 1), solver->work->g.col(NHORIZON - 1), solver->work->x1.data, 1, NSTATES);
    matmulf(solver->work->x1.data, solver->work->x2.data, solver->cache->rho, 1, NSTATES);
    matmul(solver->work->Xref.col(NHORIZON - 1), solver->cache->PinfT.data, solver->work->x1.data, 1, NSTATES, NSTATES);

    matadd(solver->work->x1.data, solver->work->x2.data, solver->work->x3.data, 1, NSTATES);
    matneg(solver->work->x3.data, solver->work->p.col(NHORIZON - 1), 1, NSTATES);
}

};
#endif //TINYMPC_ADMM_RVV_HPP
