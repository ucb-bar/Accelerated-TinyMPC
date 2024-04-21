#include <stdio.h>
#include <cstdint>

#include "tinympc/admm.hpp"

extern "C"
{

int tiny_solve(TinySolver *solver)
{
    // Initialize variables
    solver->work->status = 11; // TINY_UNSOLVED
    solver->work->iter = 1;

    forward_pass(solver);
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
    }
    return 1;
}

} /* extern "C" */
