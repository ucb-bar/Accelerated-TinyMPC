//
// Created by widyadewi on 2/23/24.
//

#pragma once
#ifndef TINYMPC_ADMM_RVV_HPP
#define TINYMPC_ADMM_RVV_HPP

#include "types_rvv.hpp"

extern "C" {

// u1 = x[:, i] * Kinf; u2 = u1 + d; u[:, i] = -u2
inline void forward_pass_1(TinySolver *solver, int i, tiny_VectorNu &u1_, tiny_VectorNu &u2_) {
    matmul(solver->work->x.col(i), solver->cache->Kinf.data, u1_.data, 1, NINPUTS, NSTATES);
    matadd(u1_.data, solver->work->d.col(i), u2_.data, 1, NINPUTS);
    matneg(u2_.data, solver->work->u.col(i), 1, NINPUTS);
    printx(solver->work->u.col(i), 1, NINPUTS, "fp1 ui");
}

// x[:, i+1] = Adyn * x[:, i] + Bdyn * u[:, i]
inline void forward_pass_2(TinySolver *solver, int i, tiny_VectorNx &x1_, tiny_VectorNx &x2_) {
    matmul(solver->work->x.col(i), solver->work->Adyn.data, x1_.data, 1, NSTATES, NSTATES);
    matmul(solver->work->u.col(i), solver->work->Bdyn.data, x2_.data, 1, NSTATES, NSTATES);
    matadd(x1_.data, x2_.data, solver->work->x.col(i + 1), 1, NSTATES);
    printx(solver->work->x.col(i + 1), 1, NSTATES, "fp2 xip1");
}

// d[:, i] = Quu_inv * (BdynT * p[:, i+1] + r[:, i]);
inline void backward_pass_1(TinySolver *solver, int i, tiny_MatrixNuNx &BdynT, tiny_VectorNu &u1_, tiny_VectorNu &u2_) {
    matmul(solver->work->p.col(i + 1), BdynT.data, u1_.data, 1, NINPUTS, NSTATES);
    matadd(solver->work->r.col(i), u1_.data, u2_.data, 1, NINPUTS);
    matmul(u2_.data, solver->cache->Quu_inv.data, solver->work->d.col(i), 1, NINPUTS, NINPUTS);

    print_array_2d(BdynT.data, NINPUTS, NSTATES, "float", "BdynT" );
    printx(solver->work->p.col(i + 1), 1, NSTATES, "bp1 pip1");
    printx(solver->work->r.col(i), 1, NINPUTS, "bp1 ri");
    printx(u1_.data, 1, NINPUTS, "bp1 u1");
    printx(u2_.data, 1, NINPUTS, "bp1 u2");
    printx(solver->work->d.col(i), 1, NINPUTS, "bp1 d");
}

// p[:, i] = q[:, i] + AmBKt * p[:, i + 1] - KinfT * r[:, i]
inline void backward_pass_2(TinySolver *solver, int i,
                            tiny_MatrixNxNu &KinfT, tiny_VectorNx &x1_, tiny_VectorNx &x2_, tiny_VectorNx &x3_) {
    matmul(solver->work->p.col(i + 1), solver->cache->AmBKt.data, x1_.data, 1, NSTATES, NSTATES);
    matmul(solver->work->r.col(i), KinfT.data, x2_.data, 1, NSTATES, NINPUTS);
    matsub(x1_.data, x2_.data, x3_.data, 1, NSTATES);
    matadd(x3_.data, solver->work->q.col(i), solver->work->p.col(i), 1, NSTATES);
    printx(solver->work->p.col(i), 1, NSTATES, "bp2 pi");
}

// y u znew  g x vnew
inline void update_dual_1(TinySolver *solver, tiny_MatrixNuNhm1 &m1_, tiny_MatrixNxNh &s1_) {
    matadd(solver->work->y.data, solver->work->u.data, m1_.data, NHORIZON - 1, NINPUTS);
    matsub(m1_.data, solver->work->znew.data, solver->work->y.data, NHORIZON - 1, NINPUTS);
    matadd(solver->work->g.data, solver->work->x.data, s1_.data, NHORIZON, NSTATES);
    matsub(s1_.data, solver->work->vnew.data, solver->work->g.data, NHORIZON, NSTATES);
}

// Box constraints on input
inline void update_slack_1(TinySolver *solver, tiny_MatrixNuNhm1 &m1_) {
    matadd(solver->work->u.data, solver->work->y.data, solver->work->znew.data, NHORIZON - 1, NINPUTS);
    if (solver->settings->en_input_bound) {
        cwisemax(solver->work->u_min.data, solver->work->znew.data, m1_.data, NHORIZON - 1, NINPUTS);
        cwisemin(solver->work->u_max.data, m1_.data, solver->work->znew.data, NHORIZON - 1, NINPUTS);
    }
}

// Box constraints on state
inline void update_slack_2(TinySolver *solver, tiny_MatrixNxNh &s1_) {
    matadd(solver->work->x.data, solver->work->g.data, solver->work->vnew.data, NHORIZON, NSTATES);
    if (solver->settings->en_state_bound) {
        cwisemax(solver->work->x_min.data, solver->work->vnew.data, s1_.data, NHORIZON, NSTATES);
        cwisemin(solver->work->x_max.data, s1_.data, solver->work->vnew.data, NHORIZON, NSTATES);
    }
}

inline void primal_residual_state(TinySolver *solver, tiny_MatrixNxNh &s1_, tiny_MatrixNxNh &s2_) {
    matsub(solver->work->x.data, solver->work->vnew.data, s1_.data, NHORIZON, NSTATES);
    cwiseabs(s1_.data, s2_.data, NHORIZON, NSTATES);
    solver->work->primal_residual_state = maxcoeff(s2_.data, NHORIZON, NSTATES);
}

inline void dual_residual_state(TinySolver *solver, tiny_MatrixNxNh &s1_, tiny_MatrixNxNh &s2_) {
    matsub(solver->work->v.data, solver->work->vnew.data, s1_.data, NHORIZON, NSTATES);
    cwiseabs(s1_.data, s2_.data, NHORIZON, NSTATES);
    solver->work->dual_residual_state = maxcoeff(s2_.data, NHORIZON, NSTATES) * solver->cache->rho;
}

inline void primal_residual_input(TinySolver *solver, tiny_MatrixNuNhm1 &m1_, tiny_MatrixNuNhm1 &m2_) {
    matsub(solver->work->u.data, solver->work->znew.data, m1_.data, NHORIZON - 1, NINPUTS);
    cwiseabs(m1_.data, m2_.data, NHORIZON - 1, NINPUTS);
    solver->work->primal_residual_input = maxcoeff(m2_.data, NHORIZON - 1, NINPUTS);
}

inline void dual_residual_input(TinySolver *solver, tiny_MatrixNuNhm1 &m1_, tiny_MatrixNuNhm1 &m2_) {
    matsub(solver->work->z.data, solver->work->znew.data, m1_.data, NHORIZON - 1, NINPUTS);
    cwiseabs(m1_.data, m2_.data, NHORIZON - 1, NINPUTS);
    solver->work->dual_residual_input = maxcoeff(m2_.data, NHORIZON - 1, NINPUTS) * solver->cache->rho;
}

inline void update_linear_cost_1(TinySolver *solver, tiny_MatrixNuNhm1 &m1_) {
    matsub(solver->work->znew.data, solver->work->y.data, m1_.data, NHORIZON - 1, NINPUTS);
    matmulf(m1_.data, solver->work->r.data, -solver->cache->rho, NHORIZON - 1, NINPUTS);
}

inline void update_linear_cost_2(TinySolver *solver, int i, tiny_VectorNx &x1_) {
    cwisemul(solver->work->Xref.col(i), solver->work->Q.data, x1_.data, 1, NSTATES);
    matmulf(x1_.data, solver->work->q.col(i), -1, 1, NSTATES);
}

inline void update_linear_cost_3(TinySolver *solver, tiny_MatrixNxNh &s1_, tiny_MatrixNxNh &s2_) {
    matsub(solver->work->vnew.data, solver->work->g.data, s1_.data, NHORIZON, NSTATES);
    matmulf(s1_.data, s2_.data, solver->cache->rho, NHORIZON, NSTATES);
    matsub(solver->work->q.data, s2_.data, s1_.data, NHORIZON, NSTATES);
    solver->work->q = s1_;
}

inline void update_linear_cost_4(TinySolver *solver,
                                 tiny_MatrixNxNx &PinfT, tiny_VectorNx &x1_, tiny_VectorNx &x2_, tiny_VectorNx &x3_) {
    matsub(solver->work->vnew.col(NHORIZON - 1), solver->work->g.col(NHORIZON - 1), x1_.data, 1, NSTATES);
    printx(solver->work->vnew.col(NHORIZON - 1), 1, NSTATES, "lc4 vnew");
    printx(solver->work->g.col(NHORIZON - 1), 1, NSTATES, "lc4 g");
    printx(x1_.data, 1, NSTATES, "lc4 x1");
    matmulf(x1_.data, x2_.data, solver->cache->rho, 1, NSTATES);
    printx(x2_.data, 1, NSTATES, "lc4 x2");
    matmul(solver->work->Xref.col(NHORIZON - 1), PinfT.data, x1_.data, 1, NSTATES, NSTATES);

    printx(solver->work->Xref.col(NHORIZON - 1), 1, NSTATES, "lc4 Xref" );
    print_array_2d(PinfT.data, NSTATES, NSTATES, "lc4", "PinfT" );
    printx(x1_.data, 1, NSTATES, "lc4 x1");
    matadd(x1_.data, x2_.data, x3_.data, 1, NSTATES);
    printx(x3_.data, 1, NSTATES, "lc4 x3");
    matneg(x3_.data, solver->work->p.col(NHORIZON - 1), 1, NSTATES);
    print_array_2d(solver->work->p.data, NHORIZON, NSTATES, "lc4", "p");
}

};
#endif //TINYMPC_ADMM_RVV_HPP
