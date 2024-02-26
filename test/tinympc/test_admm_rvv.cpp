//
// Created by widyadewi on 2/24/24.
//

#include <cstdio>
#include <cmath>
#include <cstdint>

#include "admm_rvv.hpp"

TinyCache cache;
TinyWorkspace work;
TinySettings settings;
TinySolver solver{&settings, &cache, &work};

tiny_VectorNu u1, u2;
tiny_VectorNx x1, x2, x3;
tiny_MatrixNuNhm1 m1, m2;
tiny_MatrixNxNh s1, s2;
tiny_MatrixNuNx BdynT;
tiny_MatrixNxNu KinfT;
tiny_MatrixNxNx PinfT;

extern "C" {

int main() {

    // forward pass
    cache.Kinf = 1.0;
    work.Adyn = 1.0;
    work.Bdyn = 1.0;
    work.x = 1.0;
    work.d = 1.0;

    printf("forward_pass_1\n");
    forward_pass_1(&solver, 2, u1, u2);
    print_array_2d(work.u.data, NHORIZON - 1, NINPUTS, "float", "work.u");

    printf("forward_pass_2\n");
    forward_pass_2(&solver, 2, x1, x2);
    print_array_2d(work.x.data, NHORIZON, NSTATES, "float", "work.x");

    // backward pass
    work.p = 1.0;
    work.q = 1.0;
    work.r = 1.0;
    cache.AmBKt = 1.0;
    cache.Quu_inv = 1.0;
    BdynT = 1.0;
    KinfT = 1.0;
    PinfT = 1.0;

    printf("backward_pass_1\n");
    backward_pass_1(&solver, 2, BdynT, u1, u2);
    print_array_2d(u1.data, 1, NINPUTS, "float", "work.u");

    printf("backward_pass_2\n");
    backward_pass_2(&solver, 2, KinfT, x1, x2, x3);
    print_array_2d(solver.work->p.data, NHORIZON, NSTATES, "float", "work.p");

    // update dual
    work.x = 1.0;
    work.y = 1.0;
    work.u = 1.0;
    work.znew = 4.0;

    printf("update_dual_1\n");
    update_dual_1(&solver, m1, s1);
    print_array_2d(solver.work->y.data, NHORIZON - 1, NINPUTS, "float", "work.y");
    print_array_2d(solver.work->g.data, NHORIZON, NSTATES, "float", "work.g");

    // update slack
    settings.en_input_bound = 1;
    work.y = 0.0;
    work.u = 100.0;
    work.u_min = -1.0;
    work.u_min =  1.0;

    printf("update_slack_1\n");
    update_slack_1(&solver, m1);
    print_array_2d(solver.work->znew.data, NHORIZON - 1, NINPUTS, "float", "work.znew");
    work.u = -100.0;
    update_slack_1(&solver, m1);
    print_array_2d(solver.work->znew.data, NHORIZON - 1, NINPUTS, "float", "work.znew");

    settings.en_state_bound = 1;
    work.g = 0.0;
    work.x = 100.0;
    work.x_min = -1.0;
    work.x_min =  1.0;

    printf("update_slack_2\n");
    update_slack_2(&solver, s1);
    print_array_2d(solver.work->vnew.data, NHORIZON, NSTATES, "float", "work.vnew");
    work.x = -100.0;
    print_array_2d(solver.work->vnew.data, NHORIZON, NSTATES, "float", "work.vnew");

    // residual state
    printf("residual states\n");
    cache.rho = 1.0;
    work.x = 1.0;
    work.v = 2.0;
    work.vnew = -3.0;

    primal_residual_state(&solver, s1, s2);
    printf("float prs: %f\n", work.primal_residual_state);

    dual_residual_state(&solver, s1, s2);
    printf("float drs: %f\n", work.dual_residual_state);

    // residual input
    printf("residual inputs\n");
    work.u = 1.0;
    work.z = 1.0;
    work.znew = 3.0;

    primal_residual_input(&solver, m1, m2);
    printf("float pri: %f\n", work.primal_residual_input);
    dual_residual_input(&solver, m1, m2);
    printf("float dri: %f\n", work.dual_residual_input);

    // linear cost
    work.vnew = 1.0;
    work.znew = 3.0;
    work.g = 3.0;
    work.y = 1.0;
    work.Xref = 1.0;
    work.Q = 2.0;
    work.q = 4.0;

    printf("linear_cost_1\n");
    update_linear_cost_1(&solver, m1);
    print_array_2d(work.r.data, NHORIZON - 1, NINPUTS, "float", "work.r");

    printf("linear_cost_2\n");
    update_linear_cost_2(&solver, 2, x1);
    print_array_2d(work.q.col(2), 1, NSTATES, "float", "work.q[i]");

    printf("linear_cost_3\n");
    update_linear_cost_3(&solver, s1, s2);
    print_array_2d(work.q.data, NHORIZON, NSTATES, "float", "work.q");

    printf("linear_cost_4\n");
    update_linear_cost_4(&solver, PinfT, x1, x2, x3);
    print_array_2d(work.p.col(NHORIZON - 1), 1, NSTATES, "float", "work.p[H-1]");
}

}