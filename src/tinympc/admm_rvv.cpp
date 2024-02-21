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
    for (int i = NHORIZON - 2; i >= 0; i--)
    {
        matadd(solver->work->p.col(i + 1).data, solver->work->r.col(i).data, x1.data, 1, NSTATES);
        matmul(x1.data, BdynT.data, u1.data, 1, NINPUTS, NSTATES);
        matmul(u1.data, solver->cache->Quu_inv.data, solver->work->d.col(i).data, 1, NINPUTS, NINPUTS);

        matmul(solver->work->r.col(i).data, KinfT.data, x1.data, 1, NSTATES, NINPUTS);
        matmul(solver->work->p.col(i + 1).data, solver->cache->AmBKt.data, x1.data, 1, NSTATES, NSTATES);
        matsub(x1.data, x2.data, x3.data, 1, NSTATES);
        matadd(x3.data, solver->work->q.col(i).data, solver->work->p.col(i).data, 1, NSTATES);
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
 */
void forward_pass(TinySolver *solver)
{
    for (int i = 0; i < NHORIZON - 1; i++)
    {
        print_array_2d(solver->work->x.col(i).data, 1, NSTATES, "float", "x_i");
        print_array_2d(solver->cache->Kinf.data, NINPUTS, NSTATES, "float", "Kinf");
        matmul(solver->work->x.col(i).data, solver->cache->Kinf.data, u1.data, 1, NINPUTS, NSTATES);
        print_array_2d(u1.data, 1, NINPUTS, "float", "u1");
        matadd(u1.data, solver->work->d.col(i).data, u2.data, 1, NINPUTS);
        matneg(u2.data, solver->work->u.col(i).data, 1, NINPUTS);

        matmul(solver->work->x.col(i).data, solver->work->Adyn.data, x1.data, 1, NSTATES, NSTATES);
        matmul(solver->work->u.col(i).data, solver->work->Bdyn.data, x2.data, 1, NSTATES, NSTATES);
        matadd(x1.data, x2.data, solver->work->x.col(i + 1).data, 1, NSTATES);
    }
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint projection function
 */
void update_slack(TinySolver *solver)
{
    matadd(solver->work->u.data, solver->work->y.data, solver->work->znew.data, NHORIZON - 1, NINPUTS);
    // Box constraints on input
    if (solver->settings->en_input_bound)
    {
        cwisemax(solver->work->u_min.data, solver->work->znew.data, m1.data, NHORIZON - 1, NINPUTS);
        cwisemin(solver->work->u_max.data, m1.data, solver->work->znew.data, NHORIZON - 1, NINPUTS);
    }

    matadd(solver->work->x.data, solver->work->g.data, solver->work->vnew.data, NHORIZON, NSTATES);
    // Box constraints on state
    if (solver->settings->en_state_bound)
    {
        cwisemax(solver->work->x_min.data, solver->work->vnew.data, s1.data, NHORIZON, NSTATES);
        cwisemin(solver->work->x_max.data, s1.data, solver->work->vnew.data, NHORIZON, NSTATES);
    }
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
 */
void update_dual(TinySolver *solver)
{
    matadd(solver->work->y.data, solver->work->u.data, m1.data, NHORIZON - 1, NINPUTS);
    matsub(m1.data, solver->work->znew.data, solver->work->y.data, NHORIZON - 1, NINPUTS);
    matadd(solver->work->g.data, solver->work->x.data, s1.data, NHORIZON, NSTATES);
    matsub(s1.data, solver->work->vnew.data, solver->work->g.data, NHORIZON, NSTATES);
}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
 */
void update_linear_cost(TinySolver *solver)
{
    matsub(solver->work->znew.data, solver->work->y.data, m1.data, NHORIZON - 1, NINPUTS);
    matmulf(m1.data, solver->work->r.data, -solver->cache->rho, NHORIZON - 1, NINPUTS);

    for (int i = 0; i < NHORIZON; i++) {
        cwisemul(solver->work->Xref.col(i).data, solver->work->Q.data, x1.data, 1, NSTATES);
        matmulf(x1.data, solver->work->q.col(i).data, -1, 1, NSTATES);
    }
    matsub(solver->work->vnew.data, solver->work->g.data, s1.data, NHORIZON, NSTATES);
    matmulf(s1.data, s2.data, solver->cache->rho, NHORIZON, NSTATES);
    matsub(solver->work->q.data, s2.data, s1.data, NHORIZON, NSTATES);
    solver->work->q = s1;

    matsub(solver->work->vnew.col(NHORIZON - 1).data, solver->work->g.col(NHORIZON - 1).data, x1.data, 1, NSTATES);
    matmulf(x1.data, x2.data, solver->cache->rho, 1, NSTATES);
    matmul(solver->work->Xref.col(NHORIZON - 1).data, PinfT.data, x1.data, 1, NSTATES, NSTATES);
    matadd(x1.data, x2.data, x3.data, 1, NSTATES);
    matneg(x3.data, solver->work->p.col(NHORIZON - 1).data, 1, NSTATES);
}

void tiny_init(TinySolver *solver) {

}

int tiny_solve(TinySolver *solver)
{
    // Transpose these matrices once
    transpose(solver->work->Bdyn.data, BdynT.data, NSTATES, NINPUTS);
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
        // Solve linear system with Riccati and roll out to get new trajectory
        update_primal(solver);
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
            matsub(solver->work->x.data, solver->work->vnew.data, s1.data, NHORIZON, NSTATES);
            cwiseabs(s1.data, s2.data, NHORIZON, NSTATES);
            solver->work->primal_residual_state = maxcoeff(s2.data, NHORIZON, NSTATES);

            matsub(solver->work->v.data, solver->work->vnew.data, s1.data, NHORIZON, NSTATES);
            cwiseabs(s1.data, s2.data, NHORIZON, NSTATES);
            solver->work->dual_residual_state = maxcoeff(s2.data, NHORIZON, NSTATES) * solver->cache->rho;

            matsub(solver->work->u.data, solver->work->znew.data, m1.data, NHORIZON - 1, NINPUTS);
            cwiseabs(m1.data, m2.data, NHORIZON - 1, NINPUTS);
            solver->work->primal_residual_input = maxcoeff(m2.data, NHORIZON - 1, NINPUTS);

            matsub(solver->work->z.data, solver->work->znew.data, m1.data, NHORIZON - 1, NINPUTS);
            cwiseabs(m1.data, m2.data, NHORIZON - 1, NINPUTS);
            solver->work->dual_residual_input = maxcoeff(m2.data, NHORIZON - 1, NINPUTS) * solver->cache->rho;

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
