//
// Created by widyadewi on 2/24/24.
//

#include <cstdio>
#include <cmath>
#include <cstdint>

#include "matlib/common.h"
#include "matlib/matlib_rvv.h"
#include "admm.hpp"
#include "test_admm_rvv.hpp"

#define DEBUG 0

TinyCache cache;
TinyWorkspace work;
TinySettings settings;
TinySolver solver{&settings, &cache, &work};

extern "C" {

void init_solver() {
    cache.Quu_inv.set(Quu_inv_data);
    cache.AmBKt.set(AmBKt_data);
    cache.Kinf.set(Kinf_data);
    cache.Pinf.set(Pinf_data);
    work.r.set(r_data);
    work.q.set(q_data);
    work.p.set(p_data);
    work.d.set(d_data);
    work.x.set(x_data);
    work.u.set(u_data);
    work.g.set(g_data);
    work.y.set(y_data);
    work.Adyn.set(Adyn_data);
    work.Bdyn.set(Bdyn_data);
    work.znew.set(znew_data);
    work.vnew.set(vnew_data);
    work.u_min.set(u_min_data);
    work.u_max.set(u_max_data);
    work.x_min.set(x_min_data);
    work.x_max.set(x_max_data);
}

int main() {

    tinytype checksum = 0;
    tiny_MatrixNuNx BdynT;
    tiny_MatrixNxNu KinfT;
    tiny_MatrixNxNx PinfT;

    // forward pass
    init_solver();
    forward_pass(&solver);
    checksum = work.u.checksum();
    printf("forward_pass u       : %s (%+10f %+10f)\n", float_eq(test__forward_pass__u, checksum, 1e-6) ? "pass" : "fail", test__forward_pass__u, checksum);
    if (DEBUG) print_array_2d(work.u.data, NHORIZON - 1, NINPUTS, "float", "work.u");
    checksum = work.x.checksum();
    printf("forward_pass x       : %s (%+10f %+10f)\n", float_eq(test__forward_pass__x, checksum, 1e-6) ? "pass" : "fail", test__forward_pass__x, checksum);
    if (DEBUG) print_array_2d(work.x.data, NHORIZON, NSTATES, "float", "work.x");

    // backward pass
    init_solver();
    transpose(work.Bdyn.data, BdynT.data, NSTATES, NINPUTS);
    transpose(cache.Kinf.data, KinfT.data, NINPUTS, NSTATES);
    backward_pass(&solver);
    checksum = work.p.checksum();
    printf("backward_pass p      : %s (%+10f %+10f)\n", float_eq(test__backward_pass__d, checksum, 1e-6) ? "pass" : "fail", test__backward_pass__d, checksum);
    if (DEBUG) print_array_2d(work.p.data, NHORIZON, NSTATES, "float", "work.p");
    checksum = work.d.checksum();
    printf("backward_pass d      : %s (%+10f %+10f)\n", float_eq(test__backward_pass__p, checksum, 1e-6) ? "pass" : "fail", test__backward_pass__p, checksum);
    if (DEBUG) print_array_2d(work.d.data, NHORIZON - 1, NINPUTS, "float", "work.d");

    // update primal
    init_solver();
    update_primal(&solver);
    checksum = work.u.checksum();
    printf("update_primal u      : %s (%+10f %+10f)\n", float_eq(test__update_primal__u, checksum, 1e-6) ? "pass" : "fail", test__update_primal__u, checksum);
    if (DEBUG) print_array_2d(work.u.data, NHORIZON - 1, NINPUTS, "float", "work.u");
    checksum = work.x.checksum();
    printf("update_primal x      : %s (%+10f %+10f)\n", float_eq(test__update_primal__x, checksum, 1e-6) ? "pass" : "fail", test__update_primal__x, checksum);
    if (DEBUG) print_array_2d(work.x.data, NHORIZON, NSTATES, "float", "work.x");
    checksum = work.p.checksum();
    printf("update_primal p      : %s (%+10f %+10f)\n", float_eq(test__update_primal__p, checksum, 1e-6) ? "pass" : "fail", test__update_primal__p, checksum);
    if (DEBUG) print_array_2d(work.p.data, NHORIZON, NSTATES, "float", "work.p");
    checksum = work.d.checksum();
    printf("update_primal d      : %s (%+10f %+10f)\n", float_eq(test__update_primal__d, checksum, 1e-6) ? "pass" : "fail", test__update_primal__d, checksum);
    if (DEBUG) print_array_2d(work.d.data, NHORIZON - 1, NINPUTS, "float", "work.d");

    // update slack
    init_solver();
    update_slack(&solver);
    checksum = work.znew.checksum();
    printf("update_slack znew    : %s (%+10f %+10f)\n", float_eq(test__update_slack__znew, checksum, 1e-6) ? "pass" : "fail", test__update_slack__znew, checksum);
    if (DEBUG) print_array_2d(work.znew.data, NHORIZON - 1, NINPUTS, "float", "work.znew");
    checksum = work.vnew.checksum();
    printf("update_slack vnew    : %s (%+10f %+10f)\n", float_eq(test__update_slack__vnew, checksum, 1e-6) ? "pass" : "fail", test__update_slack__vnew, checksum);
    if (DEBUG) print_array_2d(work.vnew.data, NHORIZON, NSTATES, "float", "work.vnew");

    // update dual
    init_solver();
    update_dual(&solver);
    checksum = work.y.checksum();
    printf("update_dual y        : %s (%+10f %+10f)\n", float_eq(test__update_dual__y, checksum, 1e-6) ? "pass" : "fail", test__update_dual__y, checksum);
    if (DEBUG) print_array_2d(work.y.data, NHORIZON - 1, NINPUTS, "float", "work.y");
    checksum = work.g.checksum();
    printf("update_dual g        : %s (%+10f %+10f)\n", float_eq(test__update_dual__g, checksum, 1e-6) ? "pass" : "fail", test__update_dual__g, checksum);
    if (DEBUG) print_array_2d(work.g.data, NHORIZON, NSTATES, "float", "work.g");

    // linear cost
    init_solver();
    transpose(cache.Pinf.data, PinfT.data, NSTATES, NSTATES);
    update_linear_cost(&solver);
    checksum = work.r.checksum();
    printf("update_linear_cost r : %s (%+10f %+10f)\n", float_eq(test__update_linear_cost__r, checksum, 1e-6) ? "pass" : "fail", test__update_linear_cost__r, checksum);
    if (DEBUG) print_array_2d(work.r.data, NHORIZON - 1, NINPUTS, "float", "work.r");
    checksum = work.q.checksum();
    printf("update_linear_cost q : %s (%+10f %+10f)\n", float_eq(test__update_linear_cost__q, checksum, 1e-6) ? "pass" : "fail", test__update_linear_cost__q, checksum);
    if (DEBUG) print_array_2d(work.q.col(2), 1, NSTATES, "float", "work.q[i]");
    checksum = work.p.checksum();
    printf("update_linear_cost p : %s (%+10f %+10f)\n", float_eq(test__update_linear_cost__p, checksum, 1e-6) ? "pass" : "fail", test__update_linear_cost__p, checksum);
    if (DEBUG) print_array_2d(work.p.col(NHORIZON - 1), 1, NSTATES, "float", "work.p[H-1]");
}

}


