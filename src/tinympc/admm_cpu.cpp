#include <iostream>
#include <fstream>

#include "admm.hpp"
#include "tinympc/glob_opts.hpp"
#include "tinympc/types.hpp"
#include <Eigen/Dense>

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
     * Do backward Riccati pass then forward roll out
     */
    void update_primal(TinySolver *solver)
    {
        CYCLE_CNT_WRAPPER(backward_pass_grad, solver, "update_primal_backward_pass");
        // print value of d[0]
        CYCLE_CNT_WRAPPER(forward_pass, solver, "update_primal_forward_pass");
    }

    /**
     * Update linear terms from Riccati backward pass
     */
    void backward_pass_grad(TinySolver *solver)
    {
        solver->work->P[NHORIZON - 1] = solver->work->Qf;
        solver->work->p.col(NHORIZON - 1) = solver->work->qf;
        for (int i = NHORIZON - 2; i >= 0; i--)
        {   
            tiny_MatrixNuNu Quu = (solver->work->R + solver->work->Bdyn.transpose() * solver->work->P[i+1] * solver->work->Bdyn);
            solver->work->K[i] = Quu.colPivHouseholderQr().solve(solver->work->Bdyn.transpose() * solver->work->P[i+1] * solver->work->Adyn);
            solver->cache->AmBKt = (solver->work->Adyn - solver->work->Bdyn * solver->work->K[i]).transpose();
            solver->work->P[i] = solver->work->Q + solver->work->K[i].transpose() * solver->work->R * solver->work->K[i] + solver->cache->AmBKt * solver->work->P[i+1] * solver->cache->AmBKt.transpose();
            solver->work->d.col(i) = Quu.colPivHouseholderQr().solve(solver->work->Bdyn.transpose() * solver->work->p.col(i+1) + solver->work->r.col(i));

            // print out components of d
            // std::cout << "d[" << i << "]: " << solver->work->d.col(i).transpose() << std::endl;
            // std::cout << "p[" << i+1 << "]: " << solver->work->p.col(i+1).transpose() << std::endl;
            // std::cout << "r[" << i << "]: " << solver->work->r.col(i).transpose() << std::endl;

            solver->work->p.col(i) = solver->work->q.col(i) + solver->cache->AmBKt * (solver->work->p.col(i+1) - solver->work->P[i+1] * solver->work->Bdyn * solver->work->d.col(i))
                    + solver->work->K[i].transpose() * (solver->work->R * solver->work->d.col(i) - solver->work->r.col(i));
        }
    }

    /**
     * Use LQR feedback policy to roll out trajectory
     */
    void forward_pass(TinySolver *solver)
    {
        for (int i = 0; i < NHORIZON - 1; i++)
        {
            (solver->work->u.col(i)).noalias() = -solver->work->K[i].lazyProduct(solver->work->x.col(i)) - solver->work->d.col(i);
            (solver->work->x.col(i + 1)).noalias() = solver->work->Adyn.lazyProduct(solver->work->x.col(i)) + solver->work->Bdyn.lazyProduct(solver->work->u.col(i));
        }
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
        // solver->work->r = -(solver->Uref.array().colwise() * solver->work->rf.array()); // Uref = 0 so commented out for speed up. Need to uncomment if using Uref
        solver->work->r = -solver->cache->rho * (solver->work->znew - solver->work->y);
        solver->work->q = -(solver->work->Xref.array().colwise() * solver->work->qf.array());
        (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vnew - solver->work->g);
        // TODO replace this with computed P
        solver->work->p.col(NHORIZON - 1) = -(solver->work->Xref.col(NHORIZON - 1).transpose().lazyProduct(solver->work->P[NHORIZON-1]));
        solver->work->p.col(NHORIZON - 1) -= solver->cache->rho * (solver->work->vnew.col(NHORIZON - 1) - solver->work->g.col(NHORIZON - 1));
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
                solver->work->primal_residual_state = (solver->work->x - solver->work->vnew).cwiseAbs().maxCoeff();
                solver->work->dual_residual_state = ((solver->work->v - solver->work->vnew).cwiseAbs().maxCoeff()) * solver->cache->rho;
                solver->work->primal_residual_input = (solver->work->u - solver->work->znew).cwiseAbs().maxCoeff();
                solver->work->dual_residual_input = ((solver->work->z - solver->work->znew).cwiseAbs().maxCoeff()) * solver->cache->rho;

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

            // std::cout << solver->work->primal_residual_state << std::endl;
            // std::cout << solver->work->dual_residual_state << std::endl;
            // std::cout << solver->work->primal_residual_input << std::endl;
            // std::cout << solver->work->dual_residual_input << "\n" << std::endl;
        }
        return 1;
    }

} /* extern "C" */