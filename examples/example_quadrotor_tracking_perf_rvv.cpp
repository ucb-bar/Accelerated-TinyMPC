// Quadrotor hovering example
// Make sure in glob_opts.hpp:
// - NSTATES = 12, NINPUTS=4
// - NHORIZON = anything you want
// - tinytype = float if you want to run on microcontrollers
// States: x (m), y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi
// Inputs: u1, u2, u3, u4 (motor thrust 0-1, order from Crazyflie)

// phi, theta, psi are NOT Euler angles, they are Rodiguez parameters
// check this paper for more details: https://roboticexplorationlab.org/papers/planning_with_attitude.pdf

#include <stdio.h>
#include <stdint.h>

#include <tinympc/admm.hpp>
#include <tinympc/admm_rvv.hpp>
#include "problem_data/quadrotor_20hz_params.hpp"
#include "trajectory_data/quadrotor_20hz_y_axis_line.hpp"

#define MSTATUS_VS          0x00000600
#define MSTATUS_FS          0x00006000
#define MSTATUS_XS          0x00018000

#define NUM_PERF_TESTS 10

extern "C"
{

TinyCache cache;
TinyWorkspace work;
TinySettings settings;
TinySolver solver{&settings, &cache, &work};

int main()
{
    // General state temporary variables
    printf("Entered main!\n");

    enable_vector_operations();
    uint64_t start, end;

    tiny_VectorNx v1, v2;


    // Map data from problem_data (array in row-major order)
    cache.rho = rho_value;
    cache.Kinf.set(Kinf_data);
    transpose(cache.Kinf.data, cache.KinfT.data, NINPUTS, NSTATES);
    cache.Pinf.set(Pinf_data);
    transpose(cache.Pinf.data, cache.PinfT.data, NSTATES, NSTATES);
    cache.Quu_inv.set(Quu_inv_data);
    cache.AmBKt.set(AmBKt_data);
    cache.coeff_d2p.set(coeff_d2p_data);
    work.Adyn.set(Adyn_data);
    work.Bdyn.set(Bdyn_data);
    transpose(work.Bdyn.data, work.BdynT.data, NSTATES, NINPUTS);
    work.Q.set(Q_data);
    work.R.set(R_data);

    // Valid range for inputs and states
    work.u_min = -0.583;
    work.u_max = 1 - 0.583;
    work.x_min = -5;
    work.x_max = 5;

    // Optimization states, inputs, and settings
    work.primal_residual_state = 0;
    work.primal_residual_input = 0;
    work.dual_residual_state = 0;
    work.dual_residual_input = 0;
    work.status = 0;
    work.iter = 0;
    settings.abs_pri_tol = 0.001;
    settings.abs_dua_tol = 0.001;
    settings.max_iter = 100;
    settings.check_termination = 1;
    settings.en_input_bound = 1;
    settings.en_state_bound = 1;


    // Hovering setpoint
    
    tiny_VectorNx Xref_origin;
    Matrix<tinytype, NSTATES, NTOTAL> Xref_total;
    Xref_total.set(Xref_data);

    // tinytype Xref_origin_data[NSTATES] = {
    //     0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    
    // Xref_origin.set(Xref_origin_data);
    // // Hovers at the same point until the horizon
    // for (int j = 0; j < NHORIZON; j++) {
    //     tinytype **target = { &work.Xref.data[j] };
    //     matsetv(target, Xref_origin_data, 1, NSTATES);
    //     // print_array_1d(work.Xref.data[j], NSTATES, "float", "data");
    // }


    // print_array_2d(work.Xref.data, NHORIZON, NSTATES, "float", "data");

    // current and next simulation states
    tiny_VectorNx x0, x1;
    // Initial state
    // tinytype x0_data[NSTATES] = {
    //     -3.64893626e-02,  3.70428882e-02,  2.25366379e-01, -1.92755080e-01,
    //     -1.91678221e-01, -2.21354598e-03,  9.62340916e-01, -4.09749891e-01,
    //     -3.78764621e-01,  7.50158432e-02, -6.63581815e-01,  6.71744046e-01 };
    // x0.set(x0_data);
    x0.set(Xref_data);

unsigned long total;
int i;

// Testing forward_pass_1
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    forward_pass_1(&solver, 0);
    end = read_cycles();
    total += end - start;
}
// printf("forward_pass_1: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing forward_pass_2
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    forward_pass_2(&solver, 0);
    end = read_cycles();
    total += end - start;
}
// printf("forward_pass_2: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing backward_pass_1
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    backward_pass_1(&solver, 0);
    end = read_cycles();
    total += end - start;
}
// printf("backward_pass_1: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing backward_pass_2
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    backward_pass_2(&solver, 0);
    end = read_cycles();
    total += end - start;
}
// printf("backward_pass_2: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing update_dual_1
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    update_dual_1(&solver);
    end = read_cycles();
    total += end - start;
}
// printf("update_dual_1: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing update_slack_1
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    update_slack_1(&solver);
    end = read_cycles();
    total += end - start;
}
// printf("update_slack_1: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing update_slack_2
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    update_slack_2(&solver);
    end = read_cycles();
    total += end - start;
}
// printf("update_slack_2: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing primal_residual_state
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    primal_residual_state(&solver);
    end = read_cycles();
    total += end - start;
}
// printf("primal_residual_state: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing dual_residual_state
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    dual_residual_state(&solver);
    end = read_cycles();
    total += end - start;
}
// printf("dual_residual_state: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing primal_residual_input
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    primal_residual_input(&solver);
    end = read_cycles();
    total += end - start;
}
// printf("primal_residual_input: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing dual_residual_input
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    dual_residual_input(&solver);
    end = read_cycles();
    total += end - start;
}
// printf("dual_residual_input: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing update_linear_cost_1
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    update_linear_cost_1(&solver);
    end = read_cycles();
    total += end - start;
}
// printf("update_linear_cost_1: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing update_linear_cost_2
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    update_linear_cost_2(&solver, 0);
    end = read_cycles();
    total += end - start;
}
// printf("update_linear_cost_2: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing update_linear_cost_3
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    update_linear_cost_3(&solver);
    end = read_cycles();
    total += end - start;
}
// printf("update_linear_cost_3: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);

// Testing update_linear_cost_4
total = 0;
for (i = 0; i < NUM_PERF_TESTS; i++) {
    start = read_cycles();
    update_linear_cost_4(&solver);
    end = read_cycles();
    total += end - start;
}
// printf("update_linear_cost_4: %lu\n", total / NUM_PERF_TESTS);
printf("%lu\n", total / NUM_PERF_TESTS);


    



}

} /* extern "C" */