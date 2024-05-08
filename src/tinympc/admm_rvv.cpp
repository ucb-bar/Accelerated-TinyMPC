#include <stdio.h>
#include <cstdint>

#include "tinympc/admm_rvv.hpp"
#include "Gemmini/gemmini_params.h"

extern "C"
{
    #ifdef USE_GEMMINI
    // void mvin_matrix(tinytype * data, spad_ptr_t spad_addr, int size) {
    //     gemmini_extended3_config_ld(DIM * sizeof(tinytype), 1.0, false, 0);
    //     for(int i = size; i > 0; i -= DIM) {
    //         gemmini_extended_mvin(data + (size - i), spad_addr + (size - i) / DIM, size > DIM ? DIM : size, 1);
    //     }
    // }
    void mvin_matrix(tinytype * data, spad_ptr_t spad_addr, int rows, int cols) {
        // TODO FIX NON-PADDED SIZES
        gemmini_extended3_config_ld(cols * sizeof(tinytype), 1.0, false, 0);
        for(size_t c = 0; c < cols; c+= DIM) {
            for(size_t r = 0; r < rows; r+= DIM) {
                gemmini_extended_mvin(data + r*cols + c, spad_addr + r + c*rows/DIM, DIM, DIM);
            }
        }
        // for(int i = size; i > 0; i -= DIM) {
        //     gemmini_extended_mvin(data + (size - i), spad_addr + (size - i) / DIM, size > DIM ? DIM : size, 1);
        // }
    }
    void mvin_vector(tinytype * data, spad_ptr_t spad_addr, int size) {
        gemmini_extended3_config_ld(sizeof(tinytype), 1.0, false, 0);
        for(size_t i = 0; i < size; i++) {
            gemmini_extended_mvin(data+i, spad_addr+i, 1, 1);
        }
    }
    #endif

    /*
        tiny_VectorNx Q;
        const spad_ptr_t Q_spad = Uref_spad + SPAD_ROWS(NINPUTS * (NHORIZON - 1));
        tiny_VectorNx Qf;
        const spad_ptr_t Qf_spad = Q_spad + SPAD_ROWS(NSTATES);
        tiny_VectorNu R;
        const spad_ptr_t R_spad = Qf_spad + SPAD_ROWS(NSTATES);
        tiny_MatrixNxNx Adyn;
        const spad_ptr_t Adyn_spad = R_spad + SPAD_ROWS(NINPUTS);
        tinytype * Adyn_data;
        tiny_MatrixNxNu Bdyn;
        const spad_ptr_t Bdyn_spad = Adyn_spad + SPAD_ROWS(NSTATES * NSTATES);
        tinytype * Bdyn_data;
    */

    void tiny_init(TinySolver * solver) {
        #ifdef USE_GEMMINI
        tinytype I [DIM][DIM];
        tinytype nI [DIM][DIM];
        tinytype rI [DIM][DIM];
        tinytype nrI [DIM][DIM];

        tinytype test [DIM][DIM];

        // tinytype fwd  [DIM][DIM];
        // tinytype back [DIM][DIM];
        // tinytype max  [DIM][DIM];


        for (size_t i = 0; i < DIM; i++) {
            for (size_t j = 0; j < DIM; j++)
            {
                I[i][j] = i == j;
                nI[i][j] = -(i == j);
                rI[i][j] = (i == j) * solver->cache->rho;
                nrI[i][j] = -(i == j) * solver->cache->rho;
                // fwd[i][j] = i*DIM + j;
                // back[i][j] = 15 - (i*DIM + j);
                /* The line `test[i][j] = i*DIM + j - 8.0;` is initializing the elements of the `test`
                matrix based on the row and column indices `i` and `j`. */
                // test[i][j] = i*DIM + j - 8.0;
            }
        }

        mvin_matrix((tinytype *) I, I_spad, DIM, DIM);
        mvin_matrix((tinytype *) nI, nI_spad, DIM, DIM);

        // gemmini_extended3_config_ld(sizeof(float) * DIM, 1.0, false, 0);
        // gemmini_extended2_config_st(
        //     stride=1,                 // Assuming stride of 1 for simplicity.
        //     acc_act=0,                // No activation function required for max.
        //     acc_scale=1.0,              // Assuming no scaling is needed.
        //     pool_stride=1,            
        //     pool_size=2,              
        //     pool_out_dim=DIM,  // This should match your vector length.
        //     porows=1,
        //     pocols=1,
        //     orows=1,
        //     ocols=1,
        //     upad=0,
        //     lpad=0
        // );
        // gemmini_extended2_config_st(DIM*sizeof(float), 0, 1.0, 1, 2, DIM, 1, 1, 2, 1, 0, 0);
        // gemmini_extended2_config_st(DIM*sizeof(float), 0, 1.0, 1, 2, DIM, 4, 1, 5, 4, 0, 1);
        // gemmini_extended2_config_st(DIM*sizeof(float), 0, 1.0, 1, 2, DIM, 1, 1, 2, 1, 0, 0);
        // gemmini_extended2_config_st(DIM*sizeof(float), 0, 1.0, 4, 2, DIM, 1, 1, 2, 1, 0, 0);

        // mvin_matrix((tinytype *) fwd, rI_spad, DIM, DIM);
        // mvin_matrix((tinytype *) back, nrI_spad, DIM, DIM);

        // gemmini_extended_mvin(fwd, rI_spad, DIM, 1);

        // gemmini_extended2_config_st(DIM*sizeof(float), 0, 1.0, 1, 2, DIM, 1, 1, 2, 4, 0, 1);

        // gemmini_fence();
        // gemmini_extended_mvout((float*) max, rI_spad, DIM, 1);
        // gemmini_extended_mvout((float*) max+DIM, rI_spad+1, DIM, 1);
        // gemmini_extended_mvout((float*) max+2*DIM, rI_spad+2, DIM, 1);
        // gemmini_extended_mvout((float*) max+3*DIM, rI_spad+3, DIM, 1);
        // gemmini_fence();

        // printf("Max: ");
        // for (size_t i = 0; i < DIM*DIM; i++) {
        //     printf("%0.2f ", ((float *) max)[i]);
        // }
        // printf("\n");
        // exit(0);


        // gemmini_extended_config_ex(OUTPUT_STATIONARY, RELU, 0, 1, false, false);
        // gemmini_extended_config_st(DIM*sizeof(float), 0, 1.0);

        // gemmini_extended_preload(GARBAGE_ADDR, nrI_spad, 4, 4, 4, 4);
        // gemmini_compute_preloaded(rI_spad,   I_spad);
        // gemmini_extended_mvout(nrI, nrI_spad, 4, 4);
        // gemmini_fence();

        // printf("Test_out:\n");
        // for(size_t i = 0; i < DIM; i++) {
        //     for(size_t j = 0; j < DIM; j++) {
        //         printf("%f ", nrI[i][j]);
        //     }
        //     printf("\n");
        // }
        // exit(0);


        mvin_matrix((tinytype *) rI, rI_spad, DIM, DIM);
        mvin_matrix((tinytype *) nrI, nrI_spad, DIM, DIM);
        
        // float temp_test[DIM][DIM];
        // gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
        // gemmini_extended_preload(GARBAGE_ADDR, 0x100, 4, 4, 4, 4);
        // gemmini_compute_preloaded(I_spad, nI_spad);
        // gemmini_extended_config_st(DIM*sizeof(float), 0, 1.0);
        // printf("Moving Out\n");
        // gemmini_extended_mvout(temp_test, 0x100, 4, 4);
        // gemmini_fence();
        // for (size_t i = 0; i < DIM; i++) {
        //     for (size_t j = 0; j < DIM; j++) {
        //         printf("%f ", temp_test[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("Done!\n");
        // exit(0);
        
        mvin_matrix(solver->cache->Kinf.data, Kinf_spad, NINPUTS, NSTATES);
        mvin_matrix(solver->cache->KinfT.data, KinfT_spad, NSTATES, NINPUTS);
        mvin_matrix(solver->cache->Pinf.data, Pinf_spad, NSTATES, NSTATES);
        mvin_matrix(solver->cache->Quu_inv.data, Quu_inv_spad, NINPUTS, NINPUTS);
        mvin_matrix(solver->cache->AmBKt.data, AmBKt_spad, NSTATES,  NSTATES);
        mvin_matrix(solver->cache->coeff_d2p.data, coeff_d2p_spad, NSTATES, NINPUTS);

        mvin_matrix(solver->work->Adyn.data, Adyn_spad, NSTATES, NSTATES);
        mvin_matrix(solver->work->Bdyn.data, Bdyn_spad, NSTATES, NINPUTS);
        mvin_matrix(solver->work->BdynT.data, BdynT_spad, NINPUTS, NSTATES);
        #endif 
    }

int tiny_solve(TinySolver *solver)
{
    // Initialize variables
    solver->work->status = 11; // TINY_UNSOLVED
    solver->work->iter = 1;

    CYCLE_CNT_WRAPPER(forward_pass, solver, "forward_pass");
    CYCLE_CNT_WRAPPER(update_slack, solver, "update_slack");
    CYCLE_CNT_WRAPPER(update_dual, solver, "update_dual");
    CYCLE_CNT_WRAPPER(update_linear_cost, solver, "update_linear_cost");
    for (int i = 0; i < solver->settings->max_iter; i++)
    {
        // Solve linear system with Riccati and roll out to get new trajectory
        CYCLE_CNT_WRAPPER(update_primal, solver, "update_primal");
        // Project slack variables into feasible domain
        CYCLE_CNT_WRAPPER(update_slack, solver, "update_slack");
        // Compute next iteration of dual variables
        CYCLE_CNT_WRAPPER(update_dual, solver, "update_dual");
        // Update linear control cost terms using reference trajectory, duals, and slack variables
        CYCLE_CNT_WRAPPER(update_linear_cost, solver, "update_linear_cost");

        #ifdef MEASURE_CYCLES
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        #endif
        if (solver->work->iter % solver->settings->check_termination == 0)
        {
            primal_residual_state(solver);
            dual_residual_state(solver);
            primal_residual_input(solver);
            dual_residual_input(solver);
            if (solver->work->primal_residual_state < solver->settings->abs_pri_tol &&
                solver->work->primal_residual_input < solver->settings->abs_pri_tol &&
                solver->work->dual_residual_state < solver->settings->abs_dua_tol &&
                solver->work->dual_residual_input < solver->settings->abs_dua_tol)
            {
                // Solved without error (return 0)
                solver->work->status = 1;
                return 0;
            }
        }

        // Save previous slack variables
        solver->work->v.set(solver->work->vnew.data);
        solver->work->z.set(solver->work->znew.data);

        solver->work->iter += 1;
        #ifdef MEASURE_CYCLES
        clock_gettime(CLOCK_MONOTONIC, &end);
        uint64_t timediff = (end.tv_sec - start.tv_sec)* 1e9 + (end.tv_nsec - start.tv_nsec);
        outputFile << "termination_check" << ", " << timediff << std::endl;
        #endif

        #ifdef DEBUG
        std::cout << solver->work->primal_residual_state << std::endl;
        std::cout << solver->work->dual_residual_state << std::endl;
        std::cout << solver->work->primal_residual_input << std::endl;
        std::cout << solver->work->dual_residual_input << "\n" << std::endl;
        #endif
    }
    return 1;
}

} /* extern "C" */
