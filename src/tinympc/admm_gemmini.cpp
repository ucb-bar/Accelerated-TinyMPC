#include <iostream>
#include "gemmini.h"

#include "admm.hpp"
#include "glob_opts.hpp"

#include <Eigen/Dense>
#include <vector>
#include <cstdlib>

#define DEBUG_MODULE "TINYALG"
#define UNROLLED
#define OPTIMIZED

using namespace Eigen;

extern "C"
{
    static uint64_t startTimestamp;

    static uint64_t read_cycles() {
        uint64_t cycles;
        asm volatile ("rdcycle %0" : "=r" (cycles));
        return cycles;
    }
// static void sp_tiled_matmul_ws(
//         const elem_t * A, const elem_t * B, const void * D, void * C,
//         scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
//         size_t I, size_t J, size_t K, 
//         size_t pad_I, size_t pad_J, size_t pad_K,
//         size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
//         bool a_transpose, bool b_transpose,
//         bool full_C, bool low_D,
//         bool no_bias, bool repeating_bias,
//         int act,

// static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
//         const elem_t* A, const elem_t* B,
//         const void * D, void * C,
//         size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
//         scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
//         int act, acc_scale_t scale, acc_scale_t bert_scale,
//         bool repeating_bias,
//         bool transpose_A, bool transpose_B,
//         bool full_C, bool low_D,
//         uint8_t weightA,
//         enum tiled_matmul_type_t tiled_matmul_type) {
//         int a_spad_id, int b_spad_id) {
    
    void sp_tiled_matmul_eigen(
        const Matrix<float, Dynamic, Dynamic, RowMajor>&A,
        const Matrix<float, Dynamic, Dynamic, RowMajor>&B,
        Matrix<float, Dynamic, Dynamic, RowMajor>&C,
        bool transpose_A, bool transpose_B) 
    {
            int i = transpose_A ? A.cols() : A.rows();
            int j = transpose_B ? B.rows() : B.cols();
            int k = transpose_B ? B.cols() : B.rows();
            int pad_I = 0;
            int pad_J = 3;
            int pad_K = 0;
            if(i == 1 & j == 3 & k == 3) {
                pad_I = 3;
                pad_J = 0;
            }
            // printf("Calling Tiled Matmul\n");
            printf("a: %p\tb: %p\tpre: NULL\tout: %p\t"
           "I: %zu\tJ: %zu\tK: %zu\t"
           "pad_I: 0\tpad_J: 0\tpad_K: 0\t"
           "stride_A: %zu\tstride_B: %zu\tstride_D: %zu\tstride_C: %zu\t"
           "a_transpose: %d\tb_transpose: %d\t"
           "full_C: false\tlow_D: false\t"
           "no_bias: true\trepeating_bias: false\t"
           "act: NO_ACTIVATION\t"
           "a_spad_id: 1\tb_spad_id: 1\n",
           (void *)A.data(), (void *)B.data(), (void *)C.data(),
           i, j, k,
           transpose_A ? i : k, transpose_B ? k : j, j, j,
           transpose_A, transpose_B,
           NO_ACTIVATION, 1, 1);
            sp_tiled_matmul_ws(
                A.data(), B.data(), NULL, C.data(),
                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                i, j, k, 0, 0, 0,
                transpose_A ? i : k, transpose_B ? k : j, j, j,
                transpose_A, transpose_B,
                false, false,
                true, false, // TODO no_bias??
                NO_ACTIVATION,
                1, 1
            );
            // printf("Finishing Tiled Matmul\n");
            fflush(stdout);
    }

    void tiled_matmul_outer_eigen (
        const Matrix<float, Dynamic, Dynamic, RowMajor>&A,
        const Matrix<float, Dynamic, Dynamic, RowMajor>&B,
        Matrix<float, Dynamic, Dynamic, RowMajor>&C,
        bool transpose_A, bool transpose_B) 
    {
            int i = transpose_A ? A.cols() : A.rows();
            int j = transpose_B ? B.rows() : B.cols();
            int k = transpose_B ? B.cols() : B.rows();
            int tile_I = (i + DIM - 1) / DIM;
            int tile_J = (j + DIM - 1) / DIM;
            int tile_K = (k + DIM - 1) / DIM;
            tiled_matmul_outer_simple(i, j, k,
                    A.data(), B.data(), NULL, C.data(),
                    transpose_A ? i : k, transpose_B ? k : j, j, j,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    tile_I, tile_J, tile_K,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    transpose_A, transpose_B,
                    false, false,
                    0,
                    WS
                    );
    }

    void tiled_matmul_spad_dram(
        const uint32_t sp_A_addr,
        const Matrix<float, Dynamic, Dynamic, RowMajor>&B,
        Matrix<float, Dynamic, Dynamic, RowMajor>&C,
        int i, bool transpose_A, bool transpose_B)
    {
        int j = transpose_B ? B.rows() : B.cols();
        int k = transpose_B ? B.cols() : B.rows();
        int tile_I = (i + DIM - 1) / DIM;
        int tile_J = (j + DIM - 1) / DIM;
        int tile_K = (k + DIM - 1) / DIM;

        tiled_matmul_outer_simple_dram_sp(i, j, k,
                sp_A_addr, B.data(), NULL, C.data(),
                k, j, j, j,
                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                tile_I, tile_J, tile_K,
                NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                transpose_A, transpose_B,
                false, false,
                0,
                WS
                );
    }


    void tiled_matmul_auto_eigen (
        const Matrix<float, Dynamic, Dynamic, RowMajor>&A,
        const Matrix<float, Dynamic, Dynamic, RowMajor>&B,
        Matrix<float, Dynamic, Dynamic, RowMajor>&C,
        bool transpose_A, bool transpose_B) 
    {
            int i = transpose_A ? A.cols() : A.rows();
            int j = transpose_B ? B.rows() : B.cols();
            int k = transpose_B ? B.cols() : B.rows();
            tiled_matmul_auto(i, j, k,
                    A.data(), B.data(), NULL, C.data(),
                    transpose_A ? i : k, transpose_B ? k : j, j, j,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    transpose_A, transpose_B,
                    false, false,
                    0,
                    WS
                    );
    }

    // define spad addresses for cached matrices
    // spad is row addressed and each row is 4 elements wide
    static uint32_t A_sp_addr = 0; // 144 elements, 0 to 35
    static uint32_t B_sp_addr = 36; // 48 elements, 36 to 47
    static uint32_t Kinf_sp_addr = 48; // 48 elements, 48 to 59
    static uint32_t C1_sp_addr = 60; // 16 elements, 60 to 63
    static uint32_t C2_sp_addr = 64; // 144 elements, 64 to 99
    // next available spad address is 100

    /**
     * Update linear terms from Riccati backward pass
     */
    void backward_pass_grad(TinySolver *solver)
    {
        Matrix<float, Dynamic, Dynamic, RowMajor> B_p(NINPUTS, 1);
        Matrix<float, Dynamic, Dynamic, RowMajor> dcol(NINPUTS, 1);
        Matrix<float, Dynamic, Dynamic, RowMajor> K_r(NSTATES, 1);
        Matrix<float, Dynamic, Dynamic, RowMajor> AmBKt_p(NSTATES, 1);

        for (int i = NHORIZON - 2; i >= 0; i--)
        {
            // (solver->work->d.col(i)).noalias() = solver->cache->Quu_inv * (solver->work->Bdyn.transpose() * solver->work->p.col(i + 1) + solver->work->r.col(i));
            // tiled_matmul_outer_eigen(solver->work->Bdyn, solver->work->p.col(i + 1), B_p, true, false);
            tiled_matmul_spad_dram(B_sp_addr, solver->work->p.col(i + 1), B_p, NINPUTS, true, false);

            // tiled_matmul_outer_eigen(solver->cache->Quu_inv, B_p + solver->work->r.col(i), dcol, true, false);
            tiled_matmul_spad_dram(C1_sp_addr, B_p + solver->work->r.col(i), dcol, NINPUTS, true, false);
            (solver->work->d.col(i)).noalias() = dcol;

            // (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + solver->cache->AmBKt.lazyProduct(solver->work->p.col(i + 1)) - (solver->cache->Kinf.transpose()).lazyProduct(solver->work->r.col(i)); // + solver->cache->coeff_d2p * solver->work->d.col(i); // coeff_d2p always appears to be zeros (faster to comment out)
            // tiled_matmul_outer_eigen(solver->cache->Kinf, solver->work->r.col(i), K_r, true, false);
            tiled_matmul_spad_dram(Kinf_sp_addr, solver->work->r.col(i), K_r, NSTATES, true, false);
            // tiled_matmul_outer_eigen(solver->cache->AmBKt, solver->work->p.col(i + 1), AmBKt_p, false, false);
            tiled_matmul_spad_dram(C2_sp_addr, solver->work->p.col(i + 1), AmBKt_p, NSTATES, false, false);
            (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + AmBKt_p - K_r;
        }
    }
    /**
     * Update linear terms from Riccati backward pass
     */
    void backward_pass_grad_unrolled(TinySolver *solver)
    {
        tiny_VectorNu B_p;
        tiny_VectorNu dcol;
        tiny_VectorNx K_r;
        tiny_VectorNx AmBKt_p;

        for (int i = NHORIZON - 2; i >= 0; i--)
        {
            gemmini_extended_config_ex(1, 0, 0, 1, true, false);
            gemmini_extended_config_st(4, 0, 1.000000);
            gemmini_extended3_config_ld(16, 1.000000, false, 0);
            gemmini_extended3_config_ld(4, 1.000000, false, 1);
            gemmini_extended3_config_ld(4, 1.000000, false, 2);
            gemmini_loop_ws(1, 1, 3, 0, 3, 0, solver->work->Bdyn_data, solver->work->p.col(i+1).data(), NULL, B_p.data(), 4, 1, 1, 1, true, false, false, false, false, 0, 1, 1, false);
            gemmini_fence();

            B_p += solver->work->r.col(i);

            gemmini_extended_config_ex(1, 0, 0, 1, true, false);
            gemmini_extended_config_st(4, 0, 1.000000);
            gemmini_extended3_config_ld(16, 1.000000, false, 0);
            gemmini_extended3_config_ld(4, 1.000000, false, 1);
            gemmini_extended3_config_ld(4, 1.000000, false, 2);
            gemmini_loop_ws(1, 1, 1, 0, 3, 0, solver->cache->Quu_inv_data, B_p.data(), NULL, dcol.data(), 4, 1, 1, 1, true, false, false, false, false, 0, 1, 1, false);
            gemmini_fence();

            (solver->work->d.col(i)).noalias() = dcol;

            gemmini_extended_config_ex(1, 0, 0, 1, true, false);
            gemmini_extended_config_st(4, 0, 1.000000);
            gemmini_extended3_config_ld(48, 1.000000, false, 0);
            gemmini_extended3_config_ld(4, 1.000000, false, 1);
            gemmini_extended3_config_ld(4, 1.000000, false, 2);
            gemmini_loop_ws(3, 1, 1, 0, 3, 0, solver->cache->Kinf_data, solver->work->r.col(i).data(), NULL, K_r.data(), 12, 1, 1, 1, true, false, false, false, false, 0, 1, 1, false);
            gemmini_fence();

            gemmini_extended_config_ex(1, 0, 0, 1, false, false);
            gemmini_extended_config_st(4, 0, 1.000000);
            gemmini_extended3_config_ld(48, 1.000000, false, 0);
            gemmini_extended3_config_ld(4, 1.000000, false, 1);
            gemmini_extended3_config_ld(4, 1.000000, false, 2);
            gemmini_loop_ws(3, 1, 3, 0, 3, 0, solver->cache->AmBKt_data, solver->work->p.col(i+1).data(), NULL, AmBKt_p.data(), 12, 1, 1, 1, false, false, false, false, false, 0, 1, 1, false);
            gemmini_fence();

            (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + AmBKt_p - K_r;
        }
    }

    /**
     * Update linear terms from Riccati backward pass
     */
    void backward_pass_grad_unrolled_opt(TinySolver *solver)
    {
        tiny_VectorNu B_p;
        tiny_VectorNx K_r;
        tiny_VectorNx AmBKt_p;

        gemmini_extended3_config_ld(4, 1.000000, false, 1);
        gemmini_extended3_config_ld(4, 1.000000, false, 2);
        gemmini_extended_config_st(4, 0, 1.000000);
        gemmini_extended_config_ex(1, 0, 0, 1, true, false);
        gemmini_extended3_config_ld(16, 1.000000, false, 0);
        for (int i = NHORIZON - 2; i >= 0; i--)
        {
            gemmini_loop_ws(1, 1, 3, 0, 3, 0, solver->work->Bdyn_data, solver->work->p.col(i+1).data(), NULL, B_p.data(), 4, 1, 1, 1, true, false, false, false, false, 0, 1, 1, false);

            gemmini_extended3_config_ld(48, 1.000000, false, 0);
            gemmini_loop_ws(3, 1, 1, 0, 3, 0, solver->cache->Kinf_data, solver->work->r.col(i).data(), NULL, K_r.data(), 12, 1, 1, 1, true, false, false, false, false, 0, 1, 1, false);

            gemmini_extended_config_ex(1, 0, 0, 1, false, false);
            gemmini_loop_ws(3, 1, 3, 0, 3, 0, solver->cache->AmBKt_data, solver->work->p.col(i+1).data(), NULL, AmBKt_p.data(), 12, 1, 1, 1, false, false, false, false, false, 0, 1, 1, false);
            gemmini_fence();

            B_p += solver->work->r.col(i);
            gemmini_extended_config_ex(1, 0, 0, 1, true, false);
            gemmini_extended3_config_ld(16, 1.000000, false, 0);

            gemmini_loop_ws(1, 1, 1, 0, 3, 0, solver->cache->Quu_inv_data, B_p.data(), NULL, solver->work->d.col(i).data(), 4, 1, 1, 1, true, false, false, false, false, 0, 1, 1, false);

            (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + AmBKt_p - K_r;
        }
    }

    /**
     * Use LQR feedback policy to roll out trajectory
     */
    void forward_pass(TinySolver *solver)
    {
        Matrix<float, Dynamic, Dynamic, RowMajor> Kinf_x(NINPUTS, 1);
        Matrix<float, Dynamic, Dynamic, RowMajor> A_x(NSTATES, 1);
        Matrix<float, Dynamic, Dynamic, RowMajor> B_u(NSTATES, 1);

        for (int i = 0; i < NHORIZON - 1; i++)
        {
            // tiled_matmul_outer_eigen(solver->cache->Kinf, solver->work->x.col(i), Kinf_x, false, false);
            tiled_matmul_spad_dram(Kinf_sp_addr, solver->work->x.col(i), Kinf_x, NINPUTS, false, false);

            (solver->work->u.col(i)).noalias() = -Kinf_x - solver->work->d.col(i);
            // solver->work->u.col(i) << .001, .02, .3, 4;
            // DEBUG_PRINT("u(0): %f\n", solver->work->u.col(0)(0));
            // multAdyn(solver->Ax->cache.Adyn, solver->work->x.col(i));

            // printf("calculatingx Ax \n");
            // tiled_matmul_outer_eigen(solver->work->Adyn, solver->work->x.col(i), A_x, false, false);
            tiled_matmul_spad_dram(A_sp_addr, solver->work->x.col(i), A_x, NSTATES, false, false);
            // tiled_matmul_outer_eigen(solver->work->Bdyn, solver->work->u.col(i), B_u, false, false);
            tiled_matmul_spad_dram(B_sp_addr, solver->work->u.col(i), B_u, NSTATES, false, false);
            fflush(stdout);
            (solver->work->x.col(i + 1)).noalias() = A_x + B_u;
        }
    }

    void forward_pass_unrolled(TinySolver *solver)
    {
        tiny_VectorNu Kinf_x;
        tiny_VectorNx A_x;
        tiny_VectorNx B_u;

        for (int i = 0; i < NHORIZON - 1; i++)
        {
            gemmini_extended_config_ex(1, 0, 0, 1, false, false);
            gemmini_extended_config_st(4, 0, 1.0);
            gemmini_extended3_config_ld(48, 1.000000, false, 0);
            gemmini_extended3_config_ld(4, 1.000000, false, 1);
            gemmini_extended3_config_ld(4, 1.000000, false, 2);
            gemmini_loop_ws(1, 1, 3, 0, 3, 0, solver->cache->Kinf_data, solver->work->x.col(i).data(), NULL, Kinf_x.data(), 12, 1, 1, 1, false, false, false, false, false, 0, 1, 1, false);
            gemmini_fence();

            (solver->work->u.col(i)).noalias() = -Kinf_x - solver->work->d.col(i);
            // solver->work->u.col(i) << .001, .02, .3, 4;

            gemmini_extended_config_ex(1, 0, 0, 1, false, false);
            gemmini_extended_config_st(4, 0, 1.0);
            gemmini_extended3_config_ld(48, 1.000000, false, 0);
            gemmini_extended3_config_ld(4, 1.000000, false, 1);
            gemmini_extended3_config_ld(4, 1.000000, false, 2);
            gemmini_loop_ws(3, 1, 3, 0, 3, 0, solver->work->Adyn_data, solver->work->x.col(i).data(), NULL, A_x.data(), 12, 1, 1, 1, false, false, false, false, false, 0, 1, 1, false);
            gemmini_fence();

            gemmini_extended_config_ex(1, 0, 0, 1, false, false);
            gemmini_extended_config_st(4, 0, 1.0);
            gemmini_extended3_config_ld(16, 1.000000, false, 0);
            gemmini_extended3_config_ld(4, 1.000000, false, 1);
            gemmini_extended3_config_ld(4, 1.000000, false, 2);
            gemmini_loop_ws(3, 1, 1, 0, 3, 0, solver->work->Bdyn_data, solver->work->u.col(i).data(), NULL, B_u.data(), 4, 1, 1, 1, false, false, false, false, false, 0, 1, 1, false);
            gemmini_fence();
            fflush(stdout);


            (solver->work->x.col(i + 1)).noalias() = A_x + B_u;
        }
    }

    void forward_pass_unrolled_opt(TinySolver *solver)
    {
        tiny_VectorNx B_u;

        gemmini_extended_config_ex(1, 0, 0, 1, false, false);
        gemmini_extended3_config_ld(4, 1.000000, false, 1);
        gemmini_extended3_config_ld(4, 1.000000, false, 2);
        for (int i = 0; i < NHORIZON - 1; i++)
        {
            gemmini_extended_config_st(4, 0, -1.0);
            gemmini_extended3_config_ld(48, 1.000000, false, 0);
            gemmini_loop_ws(1, 1, 3, 0, 3, 0, solver->cache->Kinf_data, solver->work->x.col(i).data(), NULL, solver->work->u.col(i).data(), 12, 1, 1, 1, false, false, false, false, false, 0, 1, 1, false);

            // solver->work->u.col(i) << .001, .02, .3, 4;

            gemmini_extended_config_st(4, 0, 1.0);
            gemmini_loop_ws(3, 1, 3, 0, 3, 0, solver->work->Adyn_data, solver->work->x.col(i).data(), NULL, solver->work->x.col(i+1).data(), 12, 1, 1, 1, false, false, false, false, false, 0, 1, 1, false);
            gemmini_fence();
            (solver->work->u.col(i)).noalias() -= solver->work->d.col(i);

            gemmini_extended3_config_ld(16, 1.000000, false, 0);
            gemmini_loop_ws(3, 1, 1, 0, 3, 0, solver->work->Bdyn_data, solver->work->u.col(i).data(), NULL, B_u.data(), 4, 1, 1, 1, false, false, false, false, false, 0, 1, 1, false);
            gemmini_fence();

            (solver->work->x.col(i + 1)).noalias() += B_u;
        }
    }


    /**
     * Do backward Riccati pass then forward roll out
     */
    void update_primal(TinySolver *solver)
    {
        #ifdef UNROLLED
        #ifdef OPTIMIZED
        backward_pass_grad_unrolled_opt(solver);
        forward_pass_unrolled_opt(solver);
        #else
        backward_pass_grad_unrolled(solver);
        forward_pass_unrolled(solver);
        #endif
        #else
        backward_pass_grad(solver);
        forward_pass(solver);
        #endif
    }

    /**
     * Project slack (auxiliary) variables into their feasible domain, defined by
     * projection functions related to each constraint
     * TODO: pass in meta information with each constraint assigning it to a
     * projection function
     */
    void update_slack(TinySolver *solver)
    {
        solver->work->znew = solver->work->u + solver->work->y;
        solver->work->vnew = solver->work->x + solver->work->g;

        // Box constraints on input
        if (solver->settings->en_input_bound)
        {
            solver->work->znew = solver->work->u_max.cwiseMin(solver->work->u_min.cwiseMax(solver->work->znew));
        }

        // Box constraints on state
        if (solver->settings->en_state_bound)
        {
            solver->work->vnew = solver->work->x_max.cwiseMin(solver->work->x_min.cwiseMax(solver->work->vnew));
        }
    }

    /**
     * Update next iteration of dual variables by performing the augmented
     * lagrangian multiplier update
     */
    void update_dual(TinySolver *solver)
    {
        solver->work->y = solver->work->y + solver->work->u - solver->work->znew;
        solver->work->g = solver->work->g + solver->work->x - solver->work->vnew;
    }

    /**
     * Update linear control cost terms in the Riccati feedback using the changing
     * slack and dual variables from ADMM
     */
    void update_linear_cost(TinySolver *solver)
    {
        // solver->work->r = -(solver->Uref.array().colwise() * solver->work->r.array()); // Uref = 0 so commented out for speed up. Need to uncomment if using Uref

        Matrix<float, Dynamic, Dynamic, RowMajor> Xref_Pinf(NSTATES, 1);

        solver->work->r = -solver->cache->rho * (solver->work->znew - solver->work->y);
        // TODO does Gemmini do component-wise multiplication?
        solver->work->q = -(solver->work->Xref.array().colwise() * solver->work->Q.array());
        (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vnew - solver->work->g);
        // solver->work->p.col(NHORIZON - 1) = -(solver->work->Xref.col(NHORIZON - 1).transpose().lazyProduct(solver->cache->Pinf));
        tiled_matmul_outer_eigen(solver->work->Xref.col(NHORIZON - 1), solver->cache->Pinf, Xref_Pinf, true, false);
        solver->work->p.col(NHORIZON - 1) = -Xref_Pinf;
        solver->work->p.col(NHORIZON - 1) -= solver->cache->rho * (solver->work->vnew.col(NHORIZON - 1) - solver->work->g.col(NHORIZON - 1));
    }

    int tiny_solve(TinySolver *solver)
    {
        /************ setup scratchpad with matrices ************/
        // move in Adyn
        gemmini_extended3_config_ld(48, 1.0, false, 0);
        for (int i = 0; i < 3; i++) {
            gemmini_extended_mvin(solver->work->Adyn_data + i*48, A_sp_addr + i*12, 12, 4);
        }
        // TODO use different configuration registers; i.e. mvin1, mvin2, etc?
        // move in Bdyn
        gemmini_extended3_config_ld(16, 1.0, false, 0);
        for (int i = 0; i < 3; i++) {
            gemmini_extended_mvin(solver->work->Bdyn_data + i*16, B_sp_addr + i*4, 4, 4);
        }
        // move in Kinf
        gemmini_extended3_config_ld(48, 1.0, false, 0);
        gemmini_extended_mvin(solver->cache->Kinf_data, Kinf_sp_addr, 12, 4);
        // move in C1 (Quu_inv in code)
        gemmini_extended3_config_ld(16, 1.0, false, 0);
        gemmini_extended_mvin(solver->cache->Quu_inv_data, C1_sp_addr, 4, 4);
        // move in C2 (AmBKt in code)
        gemmini_extended3_config_ld(48, 1.0, false, 0);
        for (int i = 0; i < 3; i++) {
            gemmini_extended_mvin(solver->cache->AmBKt_data + i*48, C2_sp_addr + i*12, 12, 4);
        }

        // Initialize variables
        solver->work->status = 11;  // TINY_UNSOLVED
        solver->work->iter = 1;
        #ifdef UNROLLED
        #ifdef OPTIMIZED
        forward_pass_unrolled_opt(solver);
        #else
        forward_pass_unrolled(solver);
        #endif
        #else
        forward_pass(solver);
        #endif

        update_slack(solver);
        update_dual(solver);
        update_linear_cost(solver);
        for (int i = 0; i < solver->settings->max_iter; i++)
        {

            // Solve linear system with Riccati and roll out to get new trajectory
            update_primal(solver);

            // Project slack variables into feasible domain
            update_slack(solver);

            // Compute next iteration of dual variables
            update_dual(solver);

            // Update linear control cost terms using reference trajectory, duals, and slack variables
            update_linear_cost(solver);

            if (solver->work->iter % solver->settings->check_termination == 0)
            {
                solver->work->primal_residual_state = (solver->work->x - solver->work->vnew).cwiseAbs().maxCoeff();
                solver->work->dual_residual_state = ((solver->work->v - solver->work->vnew).cwiseAbs().maxCoeff()) * solver->cache->rho;
                solver->work->primal_residual_input = (solver->work->u - solver->work->znew).cwiseAbs().maxCoeff();
                solver->work->dual_residual_input = ((solver->work->z - solver->work->znew).cwiseAbs().maxCoeff()) * solver->cache->rho;

                if (solver->work->primal_residual_state < solver->settings->abs_pri_tol &&
                    solver->work->primal_residual_input < solver->settings->abs_pri_tol &&
                    solver->work->dual_residual_state < solver->settings->abs_dua_tol &&
                    solver->work->dual_residual_input < solver->settings->abs_dua_tol)
                {
                    solver->work->status = 1;  // TINY_SOLVED
                    return 0; // 0 means solved with no error
                }
            }

            // Save previous slack variables
            solver->work->v = solver->work->vnew;
            solver->work->z = solver->work->znew;

            solver->work->iter += 1;

            // std::cout << solver->work->primal_residual_state << std::endl;
            // std::cout << solver->work->dual_residual_state << std::endl;
            // std::cout << solver->work->primal_residual_input << std::endl;
            // std::cout << solver->work->dual_residual_input << "\n" << std::endl;
        }
        return 1;
    }

} /* extern "C" */