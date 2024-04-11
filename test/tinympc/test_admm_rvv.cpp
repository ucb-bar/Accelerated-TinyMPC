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

template<typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
void test_assert(const char *test, float expected, Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_> &actual) {
    float sum = actual.checksum();
    printf("%-24s : %s (%2.10f %2.10f)\n", test, float_eq(expected, sum, 1e-5) ? "pass" : "fail", expected, sum);
    if (DEBUG) actual.print("float", test);
}

extern "C" {

void init_solver() {
    cache.Quu_inv.set(Quu_inv_data);
    cache.AmBKt.set(AmBKt_data);
    cache.Kinf.set(Kinf_data);
    transpose(cache.Kinf.data, cache.KinfT.data, NINPUTS, NSTATES);
    cache.Pinf.set(Pinf_data);
    transpose(cache.Pinf.data, cache.PinfT.data, NSTATES, NSTATES);
    work.r.set(r_data);
    work.q.set(q_data);
    work.p.set(p_data);
    work.d.set(d_data);
    work.x.set(x_data);
    work.u.set(u_data);
    work.g.set(g_data);
    work.y.set(y_data);
    work.Q.set(Q_data);
    work.R.set(R_data);
    work.Adyn.set(Adyn_data);
    transpose(work.Adyn.data, work.AdynT.data, NSTATES, NSTATES);
    work.Bdyn.set(Bdyn_data);
    transpose(work.Bdyn.data, work.BdynT.data, NSTATES, NINPUTS);
    work.znew.set(znew_data);
    work.vnew.set(vnew_data);
    work.u_min.set(u_min_data);
    work.u_max.set(u_max_data);
    work.x_min.set(x_min_data);
    work.x_max.set(x_max_data);
    work.Xref.set(Xref_data);
    work.Uref.set(Uref_data);
    settings.en_input_bound = 1;
    settings.en_state_bound = 1;
    work.u1 = 0.0;
    work.u2 = 0.0;
}

int main() {

    enable_vector_operations();

    // forward pass
    init_solver();
    forward_pass(&solver);
    test_assert("forward_pass u", test__forward_pass__u, work.u);
    test_assert("forward_pass x", test__forward_pass__x, work.x);

    init_solver();
    forward_pass_1(&solver, 2);
    test_assert("forward_pass_1 u", test__forward_pass_1__u, work.u);

    init_solver();
    forward_pass_2(&solver, 2);
    test_assert("forward_pass_2 x", test__forward_pass_2__x, work.x);

    // backward pass
    init_solver();
    backward_pass(&solver);
    test_assert("backward_pass d", test__backward_pass__d, work.d);
    test_assert("backward_pass p", test__backward_pass__p, work.p);

    init_solver();
    backward_pass_1(&solver, 7);
    test_assert("backward_pass_1 d", test__backward_pass_1__d, work.d);

    init_solver();
    backward_pass_2(&solver, 7);
    test_assert("backward_pass_2 p", test__backward_pass_2__p, work.p);

    // update primal
    init_solver();
    update_primal(&solver);
    test_assert("update_primal u", test__update_primal__u, work.u);
    test_assert("update_primal x", test__update_primal__x, work.x);
    test_assert("update_primal p", test__update_primal__p, work.p);
    test_assert("update_primal d", test__update_primal__d, work.d);

    // update slack
    init_solver();
    update_slack(&solver);
    test_assert("update_slack znew", test__update_slack__znew, work.znew);
    test_assert("update_slack vnew", test__update_slack__vnew, work.vnew);

    // update dual
    init_solver();
    update_dual(&solver);
    test_assert("update_dual y", test__update_dual__y, work.y);
    test_assert("update_dual g", test__update_dual__g, work.g);

    // linear cost
    init_solver();
    update_linear_cost(&solver);
    test_assert("update_linear_cost r", test__update_linear_cost__r, work.r);
    test_assert("update_linear_cost q", test__update_linear_cost__q, work.q);
    test_assert("update_linear_cost p", test__update_linear_cost__p, work.p);

    init_solver();
    update_linear_cost_1(&solver);
    test_assert("update_linear_cost_1 r", test__update_linear_cost_1__r, work.r);
    test_assert("update_linear_cost_1 q", test__update_linear_cost_1__q, work.q);
    test_assert("update_linear_cost_1 p", test__update_linear_cost_1__p, work.p);
    init_solver();
    update_linear_cost_2(&solver, 2);
    test_assert("update_linear_cost_2 r", test__update_linear_cost_2__r, work.r);
    test_assert("update_linear_cost_2 q", test__update_linear_cost_2__q, work.q);
    test_assert("update_linear_cost_2 p", test__update_linear_cost_2__p, work.p);
    init_solver();
    update_linear_cost_3(&solver);
    test_assert("update_linear_cost_3 r", test__update_linear_cost_3__r, work.r);
    test_assert("update_linear_cost_3 q", test__update_linear_cost_3__q, work.q);
    test_assert("update_linear_cost_3 p", test__update_linear_cost_3__p, work.p);
    init_solver();
    update_linear_cost_4(&solver);
    test_assert("update_linear_cost_4 r", test__update_linear_cost_4__r, work.r);
    test_assert("update_linear_cost_4 q", test__update_linear_cost_4__q, work.q);
    test_assert("update_linear_cost_4 p", test__update_linear_cost_4__p, work.p);
}

}


