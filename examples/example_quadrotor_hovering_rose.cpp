#include <iostream>

#include <tinympc/admm.hpp>
#include "problem_data/quadrotor_50hz_params_constrained.hpp"
#include "mmio.h"
#include "rose.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
/* The `#include <riscv-pk/encoding.h>` statement is including the header file `encoding.h` from the
`riscv-pk` library. This library provides functions and definitions related to the RISC-V processor
architecture, specifically for the Privileged Architecture Specification. The `encoding.h` header
file contains definitions for encoding and decoding RISC-V instructions, as well as other utility
functions for working with the RISC-V architecture. */
// #include <riscv-pk/encoding.h>

//OBS:Type of what simulator -> rose 
#define ROSE_REQ_UAV_OBS 0x16 //input: x y z, attitude (3 Rodriguez parameters), vx vy vz, and attitude rate of change (3 params)
//Action:
#define ROSE_SET_FORCE  0x20 //output: a thrust command in the range 0-1 that gets mapped to 0-65535 /

uint32_t buf[8];

void read_uav_obs(float * obs) {
  send_obs_req(ROSE_REQ_UAV_OBS);//rose->
  read_obs_rsp((void *) obs);
}


Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

extern "C"
{

    TinyCache cache;
    TinyWorkspace work;
    TinySettings settings;
    TinySolver solver{&settings, &cache, &work};

    int main()
    {
        // Map data from problem_data (array in row-major order)
        cache.rho = rho_value;
        cache.Kinf = Eigen::Map<Matrix<tinytype, NINPUTS, NSTATES, Eigen::RowMajor>>(Kinf_data);
        cache.Pinf = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Pinf_data);
        cache.Quu_inv = Eigen::Map<Matrix<tinytype, NINPUTS, NINPUTS, Eigen::RowMajor>>(Quu_inv_data);
        cache.AmBKt = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(AmBKt_data);
        cache.coeff_d2p = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(coeff_d2p_data);

        work.Adyn = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Adyn_data);
        work.Bdyn = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(Bdyn_data);
        work.Q = Eigen::Map<tiny_VectorNx>(Q_data);
        work.Qf = Eigen::Map<tiny_VectorNx>(Qf_data);
        work.R = Eigen::Map<tiny_VectorNu>(R_data);
        work.u_min = tiny_MatrixNuNhm1::Constant(-0.583);
        work.u_max = tiny_MatrixNuNhm1::Constant(1-0.583);
        work.x_min = tiny_MatrixNxNh::Constant(-5);
        work.x_max = tiny_MatrixNxNh::Constant(5);

        work.Xref = tiny_MatrixNxNh::Zero();
        work.Uref = tiny_MatrixNuNhm1::Zero();
        

        work.x = tiny_MatrixNxNh::Zero();
        work.q = tiny_MatrixNxNh::Zero();
        work.p = tiny_MatrixNxNh::Zero();
        work.v = tiny_MatrixNxNh::Zero();
        work.vnew = tiny_MatrixNxNh::Zero();
        work.g = tiny_MatrixNxNh::Zero();

        work.u = tiny_MatrixNuNhm1::Zero();
        work.r = tiny_MatrixNuNhm1::Zero();
        work.d = tiny_MatrixNuNhm1::Zero();
        work.z = tiny_MatrixNuNhm1::Zero();
        work.znew = tiny_MatrixNuNhm1::Zero();
        work.y = tiny_MatrixNuNhm1::Zero();

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

        tiny_VectorNx x0, x1; // current and next simulation states

        // Hovering setpoint
        tiny_VectorNx Xref_origin;
        Xref_origin << 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        work.Xref = Xref_origin.replicate<1, NHORIZON>(); 

        // Initial state    
        x0 << 0, 1, 0, 0.2, 0, 0, 0.1, 0, 0, 0, 0, 0;
        
        int k = 0;

        while (1)
        {
            printf("tracking error at step %2d: %.4f\n", k, (x0 - work.Xref.col(1)).norm());
            
            // 1. Update measurement
            read_uav_obs(work.x.col(0).data());

            // 2. Update reference (if needed)

            // 3. Reset dual variables (if needed)
            work.y = tiny_MatrixNuNhm1::Zero();
            work.g = tiny_MatrixNxNh::Zero();

            // 4. Solve MPC problem
            tiny_solve(&solver);
            send_action(work.u.col(0).data(), ROSE_SET_FORCE, 12);


            // std::cout << work.iter << std::endl;
            // std::cout << work.u.col(0).transpose().format(CleanFmt) << std::endl;

            // 5. Simulate forward

            // std::cout << x0.transpose().format(CleanFmt) << std::endl;
            k++;
        }

        return 0;
    }

} /* extern "C" */