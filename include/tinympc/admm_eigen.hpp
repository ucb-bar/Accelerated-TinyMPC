//
// Created by widyadewi on 2/26/24.
//

#pragma once
#ifndef TINYMPC_ADMM_EIGEN_HPP
#define TINYMPC_ADMM_EIGEN_HPP

#include "types_eigen.hpp"

extern "C" {

inline void forward_pass_1(TinySolver *solver, int i) {
    (solver->work->u.col(i)).noalias() = -solver->cache->Kinf.lazyProduct(solver->work->x.col(i)) - solver->work->d.col(i);
}

inline void forward_pass_2(TinySolver *solver, int i) {
    (solver->work->x.col(i + 1)).noalias() = solver->work->Adyn.lazyProduct(solver->work->x.col(i)) + solver->work->Bdyn.lazyProduct(solver->work->u.col(i));
}

inline void backward_pass_1(TinySolver *solver, int i) {
    (solver->work->d.col(i)).noalias() = solver->cache->Quu_inv * (solver->work->Bdyn.transpose() * solver->work->p.col(i + 1) + solver->work->r.col(i));
}

inline void backward_pass_2(TinySolver *solver, int i) {
    (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + solver->cache->AmBKt.lazyProduct(solver->work->p.col(i + 1)) - (solver->cache->Kinf.transpose()).lazyProduct(solver->work->r.col(i)); // + solver->cache->coeff_d2p * solver->work->d.col(i); // coeff_d2p always appears to be zeros (faster to comment out)
}

inline void update_slack_1(TinySolver *solver) {
    solver->work->znew = solver->work->u + solver->work->y;
    if (solver->settings->en_input_bound) {
        solver->work->znew = solver->work->u_max.cwiseMin(solver->work->u_min.cwiseMax(solver->work->znew));
    }
}

inline void update_dual_1(TinySolver *solver) {
    solver->work->y = solver->work->y + solver->work->u - solver->work->znew;
    solver->work->g = solver->work->g + solver->work->x - solver->work->vnew;
}

// Box constraints on state
inline void update_slack_2(TinySolver *solver) {
    solver->work->vnew = solver->work->x + solver->work->g;
    if (solver->settings->en_state_bound) {
        solver->work->vnew = solver->work->x_max.cwiseMin(solver->work->x_min.cwiseMax(solver->work->vnew));
    }
}

inline void primal_residual_state(TinySolver *solver) {
    solver->work->primal_residual_state = (solver->work->x - solver->work->vnew).cwiseAbs().maxCoeff();
}

inline void dual_residual_state(TinySolver *solver) {
    solver->work->dual_residual_state = ((solver->work->v - solver->work->vnew).cwiseAbs().maxCoeff()) * solver->cache->rho;
}

inline void primal_residual_input(TinySolver *solver) {
    solver->work->primal_residual_input = (solver->work->u - solver->work->znew).cwiseAbs().maxCoeff();
}

inline void dual_residual_input(TinySolver *solver) {
    solver->work->dual_residual_input = ((solver->work->z - solver->work->znew).cwiseAbs().maxCoeff()) * solver->cache->rho;
}

inline void update_linear_cost_1(TinySolver *solver) {
    solver->work->r = -solver->cache->rho * (solver->work->znew - solver->work->y);
}

inline void update_linear_cost_2(TinySolver *solver) {
    solver->work->q = -(solver->work->Xref.array().colwise() * solver->work->Q.array());
}

inline void update_linear_cost_3(TinySolver *solver) {
    (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vnew - solver->work->g);
}

inline void update_linear_cost_4(TinySolver *solver) {
    solver->work->p.col(NHORIZON - 1) = -(solver->work->Xref.col(NHORIZON - 1).transpose().lazyProduct(solver->cache->Pinf));
    solver->work->p.col(NHORIZON - 1) -= solver->cache->rho * (solver->work->vnew.col(NHORIZON - 1) - solver->work->g.col(NHORIZON - 1));
}

};
#endif //TINYMPC_ADMM_EIGEN_HPP
