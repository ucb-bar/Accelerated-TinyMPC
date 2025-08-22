#include <stdio.h>
#include <cstdint>

#include <admm.hpp>
#include <stddef.h> 


// #define MAX_ITERS 1


extern "C"
{

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
                
                #ifndef MAX_ITERS
                    // printf("Converged after %d iterations\n", solver->work->iter);
                    return 0;
                #endif
            }
        }
        // Save previous slack variables
        matsetv(solver->work->v.data, solver->work->vnew.data, solver->work->v.outer, solver->work->v.inner);
        matsetv(solver->work->z.data, solver->work->znew.data, solver->work->z.outer, solver->work->z.inner);



    }
    return 1;
}

void tiny_init(TinySolver* solver)
{
    if (!solver || !solver->work || !solver->cache || !solver->settings)
        return;

    TinyWorkspace* w = solver->work;
    TinyCache* c = solver->cache;

    // ==== Cache ====
    init_MatrixNuNx(&c->Kinf);
    init_MatrixNxNu(&c->KinfT);
    init_MatrixNxNx(&c->Pinf);
    init_MatrixNxNx(&c->PinfT);
    init_MatrixNuNu(&c->Quu_inv);
    init_MatrixNxNx(&c->AmBKt);
    init_MatrixNxNx(&c->AmBKtT);
    init_MatrixNxNu(&c->coeff_d2p);

    c->Kinf_data = c->Kinf.data;
    c->Pinf_data = c->Pinf.data;
    c->Quu_inv_data = c->Quu_inv.data;
    c->AmBKt_data = c->AmBKt.data;

    // ==== Workspace ====

    init_MatrixNxNh(&w->x);
    init_MatrixNuNhm1(&w->u);
    init_MatrixNxNh(&w->q);
    init_MatrixNuNhm1(&w->r);
    init_MatrixNxNh(&w->p);
    init_MatrixNuNhm1(&w->d);
    init_MatrixNxNh(&w->v);
    init_MatrixNxNh(&w->vnew);
    init_MatrixNuNhm1(&w->z);
    init_MatrixNuNhm1(&w->znew);
    init_MatrixNxNh(&w->g);
    init_MatrixNuNhm1(&w->y);

    init_VectorNx(&w->Q);
    init_VectorNx(&w->Qf);
    init_VectorNu(&w->R);

    init_MatrixNxNx(&w->Adyn);
    init_MatrixNxNx(&w->AdynT);
    w->Adyn_data = w->Adyn.data;

    init_MatrixNxNu(&w->Bdyn);
    init_MatrixNuNx(&w->BdynT);
    w->Bdyn_data = w->Bdyn.data;

    init_MatrixNuNhm1(&w->u_min);
    init_MatrixNuNhm1(&w->u_max);
    init_MatrixNxNh(&w->x_min);
    init_MatrixNxNh(&w->x_max);
    init_MatrixNxNh(&w->Xref);
    init_MatrixNuNhm1(&w->Uref);

    init_VectorNu(&w->Qu);
    init_VectorNu(&w->u1);
    init_VectorNu(&w->u2);
    init_VectorNx(&w->x1);
    init_VectorNx(&w->x2);
    init_VectorNx(&w->x3);

    init_MatrixNuNhm1(&w->m1);
    init_MatrixNuNhm1(&w->m2);
    init_MatrixNxNh(&w->s1);
    init_MatrixNxNh(&w->s2);

    // ==== Settings defaults (can override later) ====
    solver->settings->abs_pri_tol = 1e-3f;
    solver->settings->abs_dua_tol = 1e-3f;
    solver->settings->max_iter = 10;
    solver->settings->check_termination = 1;
    solver->settings->en_state_bound = 1;
    solver->settings->en_input_bound = 1;

    // ==== Status ====
    w->status = 0;
    w->iter = 0;
    w->primal_residual_state = 0;
    w->primal_residual_input = 0;
    w->dual_residual_state = 0;
    w->dual_residual_input = 0;
}


} /* extern "C" */
