#include <iostream>
#include <fstream>
#include <cstdint>

#include "tinympc/admm_eigen.hpp"

#define DEBUG_MODULE "TINYALG"

extern "C"
{

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
 * Update linear terms from Riccati backward pass
 */
void backward_pass(TinySolver *solver)
{
    for (int i = NHORIZON - 2; i >= 0; i--)
    {
        backward_pass_1(solver, i);
        backward_pass_2(solver, i);
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
 */
void forward_pass(TinySolver *solver)
{
    for (int i = 0; i < NHORIZON - 1; i++)
    {
        forward_pass_1(solver, i);
        forward_pass_2(solver, i);
    }
}

/**
 * Do backward Riccati pass then forward roll out
 */
void update_primal(TinySolver *solver)
{
    CYCLE_CNT_WRAPPER(backward_pass, solver, "update_primal_backward_pass");
    CYCLE_CNT_WRAPPER(forward_pass, solver, "update_primal_forward_pass");
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint
 * TODO: pass in meta information with each constraint assigning it to a
 * projection function
 */
void update_slack(TinySolver *solver)
{
    update_slack_1(solver);
    update_slack_2(solver);
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
 */
void update_dual(TinySolver *solver)
{
    update_dual_1(solver);
}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
 */
void update_linear_cost(TinySolver *solver)
{
    update_linear_cost_1(solver);
    update_linear_cost_2(solver);
    update_linear_cost_3(solver);
    update_linear_cost_4(solver);
}

void tiny_init(TinySolver *solver) {

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
                solver->work->status = 1; // TINY_SOLVED
                return 0;                 // 0 means solved with no error
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
    }
    return 1;
}

} /* extern "C" */