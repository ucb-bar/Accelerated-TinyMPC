#include <stdio.h>
#include <time.h>

#include <tinympc/admm.hpp>
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
    matsetv(x0.data, work.Xref.data[0], 1, NSTATES);

    tiny_init(&solver);

    for (int k = 0; k < NTOTAL - NHORIZON - 1; ++k)
    {
        matsub(x0.data, work.Xref.col(1), v1.data, 1, NSTATES);
        printf("tracking error: %f\n", matnorm(v1.data, 1, NSTATES));

        // 1. Update measurement
        matsetv(work.x.col(0), x0.data[0], 1, NSTATES);

        // 2. Update reference
        for (int i = 0; i < NHORIZON; i++)
            work.Xref._pdata[i] = Xref_total._pdata[i + k];

        // 3. Reset dual variables if needed
        work.y = 0.0;
        work.g = 0.0;

        // 4. Solve MPC problem
        clock_gettime(CLOCK_MONOTONIC, &start);
        tiny_solve(&solver);
        clock_gettime(CLOCK_MONOTONIC, &end);
        time1 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Time for iter %d: %f\n", k, time1);

        // 5. Simulate forward
        matmul(x0.data, work.Adyn.data, v1.data, 1, NSTATES, NSTATES);
        matmul(work.u.col(0), work.Bdyn.data, v2.data, 1, NSTATES, NINPUTS);
        matadd(v1.data, v2.data, x1.data, 1, NSTATES);
        matsetv(x0.data, x1.data[0], 1, NSTATES);
    }

    return 0;
}

} /* extern "C" */
