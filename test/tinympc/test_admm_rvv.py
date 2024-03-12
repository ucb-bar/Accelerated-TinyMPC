import numpy as np

class Solver:
    class Params:
        def __init__(self, NSTATES, NINPUTS, NHORIZON, NTOTAL):
            self.NSTATES = NSTATES
            self.NINPUTS = NINPUTS
            self.NHORIZON = NHORIZON
            self.NTOTAL = NTOTAL

    class Cache:
        def __init__(self, NSTATES, NINPUTS, rho):
            self.Quu_inv = np.ones((NINPUTS, NINPUTS))
            self.AmBKt = np.ones((NSTATES, NSTATES))
            self.C_3 = np.ones((NSTATES, NINPUTS))
            self.K_inf = np.ones((NINPUTS, NSTATES))
            self.P_inf = np.ones((NSTATES, NSTATES))

            self.rho = rho

    class Work:
        def __init__(self, NSTATES, NINPUTS, NHORIZON, NTOTAL):
            self.r = np.ones((NINPUTS, NHORIZON - 1))
            self.q = np.ones((NSTATES, NHORIZON))
            self.d = np.ones((NINPUTS, NHORIZON - 1))
            self.p = np.ones((NSTATES, NHORIZON))
            self.q = np.ones((NSTATES, NHORIZON))
            self.x = np.ones((NSTATES, NHORIZON))
            self.u = np.ones((NINPUTS, NHORIZON - 1))
            self.g = np.ones((NSTATES, NHORIZON))
            self.y = np.ones((NINPUTS, NHORIZON - 1))

            self.Adyn = np.ones((NSTATES, NSTATES))
            self.Bdyn = np.ones((NSTATES, NINPUTS))

            self.znew = np.ones((NINPUTS, NHORIZON - 1))
            self.vnew = np.ones((NSTATES, NHORIZON))

            self.u_min = np.ones((NINPUTS, NHORIZON - 1))
            self.u_max = np.ones((NINPUTS, NHORIZON - 1))

            self.x_min = np.ones((NSTATES, NHORIZON))
            self.x_max = np.ones((NSTATES, NHORIZON))

    def __init__(self, NSTATES, NINPUTS, NHORIZON, NTOTAL):
        self.params = self.Params(NSTATES, NINPUTS, NHORIZON, NTOTAL)
        self.cache = self.Cache(NSTATES, NINPUTS, rho=0)
        self.work = self.Work(NSTATES, NINPUTS, NHORIZON, NTOTAL)
        self.en_input_bound = True
        self.en_state_bound = False

# test type either "UNIT": single idx or "FULL": all idxs
def backward_pass(solver):
    for i in range(solver.params.NHORIZON - 2, -1, -1):
        solver.work.d[:, i] = solver.cache.Quu_inv @ (solver.work.Bdyn.T @ solver.work.p[:, i+1] + solver.work.r[:, i])
        solver.work.p[:, i] = solver.work.q[:, i] + solver.cache.AmBKt @ solver.work.p[:, i+1] - solver.cache.K_inf.T @ solver.work.r[:, i] # + C_3 @ solver.work.d[i]

def forward_pass_1(solver, i):
    solver.work.u[:, i] = solver.cache.K_inf @ solver.work.x[:, i] - solver.work.d[:, i]

def forward_pass_2(solver, i):
    solver.work.x[:, i + 1] = solver.work.Adyn @ solver.work.x[:, i] + solver.work.Bdyn @ solver.work.u[:, i]

def forward_pass(solver):
    for i in range(0, solver.params.NHORIZON - 1):
        forward_pass_1(solver, i)
        forward_pass_2(solver, i)

def update_primal(solver):
    backward_pass(solver)
    forward_pass(solver)

def update_slack(solver):
    solver.work.znew = solver.work.u + solver.work.y;
    solver.work.vnew = solver.work.x + solver.work.g;
    solver.work.znew = np.minimum(solver.work.u_max, np.maximum(solver.work.u_min, solver.work.znew))
    
    solver.work.znew = solver.work.u + solver.work.y;
    solver.work.vnew = solver.work.x + solver.work.g;
    solver.work.vnew = np.minimum(solver.work.x_max, np.maximum(solver.work.x_min, solver.work.vnew))

def update_dual(solver):
    solver.work.y = solver.work.y + solver.work.u - solver.work.znew
    solver.work.g = solver.work.g + solver.work.x - solver.work.vnew

def update_linear_cost(solver):
    solver.work.r = -solver.cache.rho * (solver.work.znew - solver.work.y)
    solver.work.q = -(solver.work.Xref @ solver.work.Q)
    solver.work.q -= solver.cache.rho * (solver.work.vnew - solver.work.g)
    solver.work.p[:, solver.params.NHORIZON - 1] = -(solver.work.Xref[solver.params.NHORIZON].T @ solver.cache.Pinf)
    solver.work.p[:, solver.params.NHORIZON - 1] -= solver.cache.rho * (solver.work.vnew[:, solver.params.NHORIZON - 1] - solver.work.g[:, solver.params.NHORIZON - 1])


def run_tests():
    ##### TEST 1: FORWARD_PASS #####
    solver = Solver(12, 4, 10, 301)
    # forward_pass(solver)
    forward_pass_1(solver, 2)
    # Checksums:
    print("FORWARD_PASS")
    print(f"Checksum u: \t {np.sum(solver.work.u)}")
    print(f"Checksum x: \t {np.sum(solver.work.x)}")
    print(f"Checksum *: \t {np.sum(solver.work.u) + np.sum(solver.work.x)}")
    # forward_pass(solver)

    
    ##### TEST 2: BACKWARD_PASS #####
    solver = Solver(12, 4, 10, 301)
    forward_pass(solver)
    # Checksums:
    print("BACKWARD_PASS")
    print(f"Checksum d: \t {np.sum(solver.work.d)}")
    print(f"Checksum p: \t {np.sum(solver.work.p)}")
    print(f"Checksum *: \t {np.sum(solver.work.d) + np.sum(solver.work.p)}")


    ##### TEST 3: UPDATE_PRIMAL #####   # multiply forward u, x by 0.01 before doing backward for overflow reason
    solver = Solver(12, 4, 10, 301)
    update_primal(solver)
    # Checksums:
    print("UPDATE_PRIMAL: (u, x * 0.01)")
    print(f"Checksum u: \t {np.sum(solver.work.u) * 0.01}")
    print(f"Checksum x: \t {np.sum(solver.work.x) * 0.01}")
    print(f"Checksum d: \t {np.sum(solver.work.d)}")
    print(f"Checksum p: \t {np.sum(solver.work.p)}")
    print(f"Checksum *: \t {np.sum(solver.work.u) * 0.01 + np.sum(solver.work.x) * 0.01 + np.sum(solver.work.d) + np.sum(solver.work.p)}")

    ##### TEST 4: UPDATE_SLACK #####
    solver = Solver(12, 4, 10, 301)
    update_slack(solver)
    # Checksums:
    print("UPDATE_SLACK")
    print(f"Checksum znew: \t {np.sum(solver.work.znew)}")
    print(f"Checksum vnew: \t {np.sum(solver.work.vnew)}")
    print(f"Checksum *: \t {np.sum(solver.work.znew) + np.sum(solver.work.vnew)}")

    ##### TEST 5: UPDATE_DUAL #####
    solver = Solver(12, 4, 10, 301)
    update_dual(solver)
    # Checksums:
    print("UPDATE_DUAL")
    print(f"Checksum y: \t {np.sum(solver.work.y)}")
    print(f"Checksum g: \t {np.sum(solver.work.g)}")
    print(f"Checksum *: \t {np.sum(solver.work.y) + np.sum(solver.work.g)}")

    ##### TEST 6: UPDATE_LINEAR_COST #####
    solver = Solver(12, 4, 10, 301)
    forward_pass(solver)
    # Checksums:
    print("UPDATE_LINEAR_COST")
    print(f"Checksum r: \t {np.sum(solver.work.r)}")
    print(f"Checksum q: \t {np.sum(solver.work.q)}")
    print(f"Checksum p: \t {np.sum(solver.work.p)}")
    print(f"Checksum *: \t {np.sum(solver.work.r) + np.sum(solver.work.q) + np.sum(solver.work.p)}")

if __name__ == "__main__":
    run_tests()