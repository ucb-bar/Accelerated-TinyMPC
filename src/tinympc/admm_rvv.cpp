#include <stdio.h>
#include <cstdint>

#include "tinympc/admm.hpp"

#define DEBUG_MODULE "TINYALG"

extern "C"
{

// Temporary variables
tiny_VectorNu u1, u2;
tiny_VectorNx x1, x2, x3;
tiny_MatrixNuNhm1 m1, m2;
tiny_MatrixNxNh s1, s2;
tiny_MatrixNuNx BdynT;
tiny_MatrixNxNu KinfT;
tiny_MatrixNxNx PinfT;

static uint64_t startTimestamp;
#ifdef MEASURE_CYCLES
std::ofstream outputFile("cycle_output.csv");
#define CYCLE_CNT_WRAPPER(func, arg, name) \
    do { \
        struct timespec start, end; \
        clock_gettime(CLOCK_MONOTONIC, &start); \
        func(arg); \
        clock_gettime(CLOCK_MONOTONIC, &end); \
        uint64_t timediff = (end.tv_sec - start.tv_sec)* 1e9 + (end.tv_nsec - start.tv_nsec); \
        outputFile << name << ", " << timediff << std::endl; \
    } while(0)
#else
#define CYCLE_CNT_WRAPPER(func, arg, name) func(arg)
#endif

/**
 * Do backward Riccati pass then forward roll out
 */
void update_primal(TinySolver *solver)
{
    CYCLE_CNT_WRAPPER(backward_pass_grad, solver, "update_primal_backward_pass");
    CYCLE_CNT_WRAPPER(forward_pass, solver, "update_primal_forward_pass");
}

/**
 * Update linear terms from Riccati backward pass
 */
void backward_pass_grad(TinySolver *solver)
{
    for (int i = NHORIZON - 2; i >= 0; i--) {
        backward_pass_1(solver, i, BdynT, u1, u2);
        backward_pass_2(solver, i, KinfT, x1, x2, x3);
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
 */

void forward_pass(TinySolver *solver)
{
    for (int i = 0; i < NHORIZON - 1; i++) {
        forward_pass_1(solver, i, u1, u2);
        forward_pass_2(solver, i, x1, x2);
    }
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint projection function
 */
void update_slack(TinySolver *solver)
{
    update_slack_1(solver, m1);
    update_slack_2(solver, s1);
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
 */
void update_dual(TinySolver *solver)
{
    update_dual_1(solver, m1, s1);
}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
 */
void update_linear_cost(TinySolver *solver)
{
    update_linear_cost_1(solver, m1);
    for (int i = 0; i < NHORIZON; i++) {
        update_linear_cost_2(solver, i, x1);
    }
    update_linear_cost_3(solver, s1, s2);
    update_linear_cost_4(solver, PinfT, x1, x2, x3);
}

void tiny_init(TinySolver *solver) {

}

int tiny_solve(TinySolver *solver)
{
    u1 = 0.0;
    u2 = 0.0;

    // Transpose these matrices once
    transpose(solver->work->Bdyn.data, BdynT.data, NSTATES, NINPUTS);
    print_array_2d(solver->work->Bdyn.data, NSTATES, NINPUTS, "float", "Bdyn" );
    print_array_2d(BdynT.data, NINPUTS, NSTATES, "float", "BdynT" );
    transpose(solver->cache->Kinf.data, KinfT.data, NINPUTS, NSTATES);
    transpose(solver->cache->Pinf.data, PinfT.data, NSTATES, NSTATES);

    // Initialize variables
    solver->work->status = 11; // TINY_UNSOLVED
    solver->work->iter = 1;

    CYCLE_CNT_WRAPPER(forward_pass, solver, "forward_pass");
    CYCLE_CNT_WRAPPER(update_slack, solver, "update_slack");
    CYCLE_CNT_WRAPPER(update_dual, solver, "update_dual");
    CYCLE_CNT_WRAPPER(update_linear_cost, solver, "update_linear_cost");
    for (int i = 0; i < solver->settings->max_iter; i++)
    {
        printf("%d ------------------------\n", i);
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
            primal_residual_state(solver, s1, s2);
            dual_residual_state(solver, s1, s2);
            primal_residual_input(solver, m1, m2);
            dual_residual_input(solver, m1, m2);
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
        solver->work->v = solver->work->vnew;
        solver->work->z = solver->work->znew;

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
