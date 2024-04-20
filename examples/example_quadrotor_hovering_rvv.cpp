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

#include <tinympc/admm.hpp>
#include "problem_data/quadrotor_50hz_params_unconstrained.hpp"

extern "C"
{

TinyCache cache;
TinyWorkspace work;
TinySettings settings;
TinySolver solver{&settings, &cache, &work};

int main()
{
    enable_vector_operations();

    // General state temporary variables
    printf("Entered main!\n");
    tiny_VectorNx v1, v2;

    // Map array from problem_data (array in row-major order)
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
    tinytype Xref_origin_data[NSTATES] = {
        0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    Xref_origin.set(Xref_origin_data);
    // Hovers at the same point until the horizon
    for (int j = 0; j < NHORIZON; j++) {
        tinytype *target = work.Xref.col(j);
        matsetv(target, Xref_origin_data, 1, NSTATES);
    }

    // current and next simulation states
    tiny_VectorNx x0, x1;
    // Initial state
    tinytype x0_data[NSTATES] = {
        -3.64893626e-02,  3.70428882e-02,  2.25366379e-01, -1.92755080e-01,
        -1.91678221e-01, -2.21354598e-03,  9.62340916e-01, -4.09749891e-01,
        -3.78764621e-01,  7.50158432e-02, -6.63581815e-01,  6.71744046e-01 };
    x0.set(x0_data);

    printf("Step,TrackingError,x,y,z,phi,theta,psi,dx,dy,dz,dphi,dtheta,dpsi,u1,u2,u3,u4\n");

    for (int k = 0; k < 70; ++k) {

        // 1. Update measurement
        // an alternative method is to use work.x.setCol(x0.data[0], 0);
        matsetv(work.x.col(0), x0.data, 1, NSTATES);

        // 2. Update reference (if needed)

        // 3. Reset dual variables (if needed)
        work.y = 0.0;
        work.g = 0.0;

        // 4. Solve MPC problem
        tiny_solve(&solver);

        // 5. Simulate forward
        // calculate x1 = work.Adyn * x0 + work.Bdyn * work.u.col(0);
        #ifdef USE_MATVEC
        matvec(work.Adyn.data, x0.data, v1.data, NSTATES, NSTATES);
        matvec(work.Bdyn.data, work.u.col(0), v2.data, NSTATES, NINPUTS);
        #else
        matmul(x0.data, work.Adyn.data, v1.data, 1, NSTATES, NSTATES);
        matmul(work.u.col(0), work.Bdyn.data, v2.data, 1, NSTATES, NINPUTS);
        #endif
        matadd(v1.data, v2.data, x0.data, 1, NSTATES);

        printf("%d,", k);

        // Print states array to CSV file
        // calculate the value of (x0 - work.Xref.col(1)).norm()
        matsub(x0.data, work.Xref.col(1), v1.data, 1, NSTATES);
        float norm = matnorm(v1.data, 1, NSTATES);
        printf("%0.7f,", norm);
        for (int i = 0; i < NSTATES; ++i) {
            printf("%0.7f", work.x(0, i));
            if (i < NSTATES - 1) {
                printf(",");
            }
        }
        printf(",");

        // Print forces to CSV file
        for (int i = 0; i < NINPUTS; ++i) {
            printf("%0.7f", work.u(i, 0));
            if (i < NINPUTS - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

} /* extern "C" */
