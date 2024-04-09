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
// #include <RoSE/encoding.h>

#include <tinympc/admm.hpp>
#include <matlib/common.h>
#include "problem_data/quadrotor_20hz_params.hpp"
#include "trajectory_data/quadrotor_20hz_y_axis_line.hpp"

extern "C"
{

TinyCache cache;
TinyWorkspace work;
TinySettings settings;
TinySolver solver{&settings, &cache, &work};


struct timespec start, end;
double time1;

int main()
{
    // General state temporary variables
    printf("Entered main!\n");

    enable_vector_operations();
    uint64_t start, end;

    // Current and next simulation states
    tiny_VectorNx x0, x1;
    tiny_VectorNx v1, v2;
    Matrix<tinytype, NTOTAL, NSTATES, RowMajor> Xref_total_raw;
    Matrix<tinytype, NSTATES, NTOTAL, ColMajor> Xref_total;

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
    work.Qf.set(Qf_data);
    work.R.set(R_data);

    // Valid range for inputs and states
    work.u_min = -0.5;
    work.u_max = 1 - 0.5;
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

    // Map data from trajectory_data
    Xref_total_raw.set(Xref_data);
    transpose(Xref_total_raw.data, Xref_total.data, NTOTAL, NSTATES);
    for (int i = 0; i < NHORIZON; i++)
        work.Xref._pdata[i] = Xref_total._pdata[i + 0];

    // Initial state
    x0.set(Xref_data);

    tiny_init(&solver);

    for (int k = 0; k < NTOTAL - NHORIZON - 1; ++k)
    {
        // Print states data to CSV file
        // calculate the value of (x0 - work.Xref.col(1)).norm()
        matsub(x0.data, work.Xref.col(1), v1.data, 1, NSTATES);
        printf("tracking error: %5.7f\n", matnorm(v1.data, 1, NSTATES));

        // 1. Update measurement
        matsetv(work.x.col(0), x0.data[0], 1, NSTATES);

        // 2. Update reference
        for (int i = 0; i < NHORIZON; i++)
            work.Xref._pdata[i] = Xref_total._pdata[i + k];

        // 3. Reset dual variables if needed
        work.y = 0.0;
        work.g = 0.0;

        // 4. Solve MPC problem
        start = read_cycles();
        tiny_solve(&solver);
        end = read_cycles();
        printf("Time for iter %d: %d\n", k, end-start);

        // 5. Simulate forward
        // calculate x1 = work.Adyn * x0 + work.Bdyn * work.u.col(0);
        matmul(x0.data, work.Adyn.data, v1.data, 1, NSTATES, NSTATES);
        matmul(work.u.col(0), work.Bdyn.data, v2.data, 1, NSTATES, NINPUTS);
        matadd(v1.data, v2.data, x0.data, 1, NSTATES);

    }
}

} /* extern "C" */
