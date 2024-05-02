//
// Created by widyadewi on 2/24/24.
//

#include <cstdio>
#include <cmath>
#include <cstdint>

#include <admm_eigen.hpp>

TinyCache cache;
TinyWorkspace work;
TinySettings settings;
TinySolver solver{&settings, &cache, &work};

extern "C" {

int main() {

    // forward pass
    cache.Kinf.setConstant(1.0);
    work.Adyn.setConstant(1.0);
    work.Bdyn.setConstant(1.0);
    work.x.setConstant(1.0);
    work.d.setConstant(1.0);

    printf("forward_pass_1\n");
    forward_pass_1(&solver, 2);
    // print_array_2d(work.u.array, NHORIZON - 1, NINPUTS, "float", "work.u");

    printf("forward_pass_2\n");
    forward_pass_2(&solver, 2);
    // print_array_2d(work.x.array, NHORIZON, NSTATES, "float", "work.x");

    // backward pass
    work.p.setConstant(1.0);
    work.q.setConstant(1.0);
    work.r.setConstant(1.0);
    cache.AmBKt.setConstant(1.0);
    cache.Quu_inv.setConstant(1.0);
    cache.Kinf.setConstant(1.0);
    cache.Pinf.setConstant(1.0);
    work.Bdyn.setConstant(1.0);

    printf("backward_pass_1\n");
    backward_pass_1(&solver, 2);
    // print_array_2d(u1.array, 1, NINPUTS, "float", "work.u");

    printf("backward_pass_2\n");
    backward_pass_2(&solver, 2);
    // print_array_2d(solver.work->p.array, NHORIZON, NSTATES, "float", "work.p");

    // update dual
    work.x.setConstant(1.0);
    work.y.setConstant(1.0);
    work.u.setConstant(1.0);
    work.znew.setConstant(4.0);

    printf("update_dual_1\n");
    update_dual_1(&solver);
    // print_array_2d(solver.work->y.array, NHORIZON - 1, NINPUTS, "float", "work.y");
    // print_array_2d(solver.work->g.array, NHORIZON, NSTATES, "float", "work.g");

    // update slack
    settings.en_input_bound = 1;
    work.y.setConstant(0.0);
    work.u.setConstant(100.0);
    work.u_min.setConstant(-1.0);
    work.u_min.setConstant(1.0);

    printf("update_slack_1\n");
    update_slack_1(&solver);
    // print_array_2d(solver.work->znew.array, NHORIZON - 1, NINPUTS, "float", "work.znew");
    work.u.setConstant(-100.0);
    update_slack_1(&solver);
    // print_array_2d(solver.work->znew.array, NHORIZON - 1, NINPUTS, "float", "work.znew");

    settings.en_state_bound = 1;
    work.g.setConstant(0.0);
    work.x.setConstant(100.0);
    work.x_min.setConstant(-1.0);
    work.x_min.setConstant(1.0);

    printf("update_slack_2\n");
    update_slack_2(&solver);
    // print_array_2d(solver.work->vnew.array, NHORIZON, NSTATES, "float", "work.vnew");
    work.x.setConstant(-100.0);
    // print_array_2d(solver.work->vnew.array, NHORIZON, NSTATES, "float", "work.vnew");

    // residual state
    printf("residual states\n");
    cache.rho = 1.0;
    work.x.setConstant(1.0);
    work.v.setConstant(2.0);
    work.vnew.setConstant(-3.0);

    primal_residual_state(&solver);
    // printf("float prs: %f\n", work.primal_residual_state);

    dual_residual_state(&solver);
    // printf("float drs: %f\n", work.dual_residual_state);

    // residual input
    printf("residual inputs\n");
    work.u.setConstant(1.0);
    work.z.setConstant(1.0);
    work.znew.setConstant(3.0);

    primal_residual_input(&solver);
    // printf("float pri: %f\n", work.primal_residual_input);
    dual_residual_input(&solver);
    // printf("float dri: %f\n", work.dual_residual_input);

    // linear cost
    work.vnew.setConstant(1.0);
    work.znew.setConstant(3.0);
    work.g.setConstant(3.0);
    work.y.setConstant(1.0);
    work.Xref.setConstant(1.0);
    work.Q.setConstant(2.0);
    work.q.setConstant(4.0);

    printf("linear_cost_1\n");
    update_linear_cost_1(&solver);
    // print_array_2d(work.r.array, NHORIZON - 1, NINPUTS, "float", "work.r");

    printf("linear_cost_2\n");
    update_linear_cost_2(&solver);
    // print_array_2d(work.q.col(2), 1, NSTATES, "float", "work.q[i]");

    printf("linear_cost_3\n");
    update_linear_cost_3(&solver);
    // print_array_2d(work.q.array, NHORIZON, NSTATES, "float", "work.q");

    printf("linear_cost_4\n");
    update_linear_cost_4(&solver);
    // print_array_2d(work.p.col(NHORIZON - 1), 1, NSTATES, "float", "work.p[H-1]");
}

}