import os
import sys
import re
import numpy as np

FILE = None
DATA = True


def create_binary_matrix(name, row, col, nul='0.01 ', one='0.02 '):
    global FILE
    fmt = f"{{0:0{col}b}}"
    full_binary = ""
    for i in range(row):
        binary = fmt.format(i).replace('1', 'x').replace('0', 'y').replace('x',nul).replace('y',one)
        full_binary += binary
    num_binary = [float(i) for i in full_binary.split(' ')[:-1]]
    arr = np.array(num_binary).reshape((row, col))
    c_arr = f"tinytype " + name + f"_data[{row} * {col}]" + " = { " + str(num_binary)[1:-1] + " };\n\n"
    if DATA:
        FILE.write(c_arr)
    return arr


class Solver:

    class Params:
        def __init__(self, NSTATES, NINPUTS, NHORIZON):
            self.NSTATES = NSTATES
            self.NINPUTS = NINPUTS
            self.NHORIZON = NHORIZON

    class Cache:
        def __init__(self, NSTATES, NINPUTS, rho):
            self.rho = rho
            self.Quu_inv = create_binary_matrix("Quu_inv", NINPUTS, NINPUTS)
            self.AmBKt = create_binary_matrix("AmBKt", NSTATES, NSTATES)
            self.Kinf = create_binary_matrix("Kinf", NINPUTS, NSTATES)
            self.Pinf = create_binary_matrix("Pinf", NSTATES, NSTATES)

    class Work:
        def __init__(self, NSTATES, NINPUTS, NHORIZON):
            self.r = create_binary_matrix("r", NINPUTS, NHORIZON - 1)
            self.q = create_binary_matrix("q", NSTATES, NHORIZON)
            self.p = create_binary_matrix("p", NSTATES, NHORIZON)
            self.d = create_binary_matrix("d", NINPUTS, NHORIZON - 1)
            self.x = create_binary_matrix("x", NSTATES, NHORIZON)
            self.u = create_binary_matrix("u", NINPUTS, NHORIZON - 1)
            self.g = create_binary_matrix("g", NSTATES, NHORIZON)
            self.y = create_binary_matrix("y", NINPUTS, NHORIZON - 1)
            self.Adyn = create_binary_matrix("Adyn", NSTATES, NSTATES)
            self.Bdyn = create_binary_matrix("Bdyn", NSTATES, NINPUTS)
            self.znew = create_binary_matrix("znew", NINPUTS, NHORIZON - 1)
            self.vnew = create_binary_matrix("vnew", NSTATES, NHORIZON)
            self.u_min = create_binary_matrix("u_min", NINPUTS, NHORIZON - 1)
            self.u_max = create_binary_matrix("u_max", NINPUTS, NHORIZON - 1)
            self.x_min = create_binary_matrix("x_min", NSTATES, NHORIZON)
            self.x_max = create_binary_matrix("x_max", NSTATES, NHORIZON)

    def __init__(self, NSTATES, NINPUTS, NHORIZON):
        self.params = self.Params(NSTATES, NINPUTS, NHORIZON)
        self.cache = self.Cache(NSTATES, NINPUTS, rho=0)
        self.work = self.Work(NSTATES, NINPUTS, NHORIZON)
        self.en_input_bound = True
        self.en_state_bound = True


def backward_pass_1(solver, i):
    solver.work.d[:, i] = solver.cache.Quu_inv @ (solver.work.Bdyn.T @ solver.work.p[:, i+1] + solver.work.r[:, i])


def backward_pass_2(solver, i):
    solver.work.p[:, i] = solver.work.q[:, i] + solver.cache.AmBKt @ solver.work.p[:, i+1] - \
                          solver.cache.Kinf.T @ solver.work.r[:, i]  # + C_3 @ solver.work.d[i]


def backward_pass(solver):
    for i in range(solver.params.NHORIZON - 2, -1, -1):
        backward_pass_1(solver, i)
        backward_pass_2(solver, i)


def forward_pass_1(solver, i):
    solver.work.u[:, i] = - solver.cache.Kinf @ solver.work.x[:, i] - solver.work.d[:, i]


def forward_pass_2(solver, i):
    solver.work.x[:, i + 1] = solver.work.Adyn @ solver.work.x[:, i] + solver.work.Bdyn @ solver.work.u[:, i]


def forward_pass(solver):
    for i in range(0, solver.params.NHORIZON - 1):
        forward_pass_1(solver, i)
        forward_pass_2(solver, i)


def update_primal(solver):
    backward_pass(solver)
    forward_pass(solver)


def update_slack_1(solver):
    solver.work.znew = solver.work.u + solver.work.y;
    if solver.en_input_bound:
        solver.work.znew = np.minimum(solver.work.u_max, np.maximum(solver.work.u_min, solver.work.znew))
 
 
def update_slack_2(solver):
    solver.work.vnew = solver.work.x + solver.work.g;
    if solver.en_state_bound:
        solver.work.vnew = np.minimum(solver.work.x_max, np.maximum(solver.work.x_min, solver.work.vnew))


def update_slack(solver):
    update_slack_1(solver)
    update_slack_2(solver)


def update_dual(solver):
    solver.work.y = solver.work.y + solver.work.u - solver.work.znew
    solver.work.g = solver.work.g + solver.work.x - solver.work.vnew


def update_linear_cost_1(solver):
    solver.work.r = -solver.cache.rho * (solver.work.znew - solver.work.y)


def update_linear_cost_2(solver, i):
    solver.work.q[:, i] = -(solver.work.Xref[:, i] * solver.work.Q.T)


def update_linear_cost_3(solver):
    solver.work.q -= solver.cache.rho * (solver.work.vnew - solver.work.g)


def update_linear_cost_4(solver):
    solver.work.p[:, solver.params.NHORIZON - 1] = -(solver.work.Xref[solver.params.NHORIZON].T @ solver.cache.Pinf)
    solver.work.p[:, solver.params.NHORIZON - 1] -= solver.cache.rho * (solver.work.vnew[:, solver.params.NHORIZON - 1]
                                                                        - solver.work.g[:, solver.params.NHORIZON - 1])


def update_linear_cost(solver):
    update_linear_cost_1(solver)
    for i in range(solver.params.NHORIZON):
        update_linear_cost_2(solver, i)
    update_linear_cost_3(solver)
    update_linear_cost_4(solver)


def print_checksum(test_name, matrix):
    global FILE
    checksum = np.sum(matrix)
    # print(f"{test_name:50}: \t {checksum:20}")
    FILE.write(f"tinytype {test_name} = {checksum};\n\n")


def run_tests():

    global FILE, DATA
    file_name = re.sub(r"\.py$", ".hpp", os.path.abspath(sys.argv[0]).replace(r'.py', '.hpp'))
    FILE = open(file_name, "w")

    # TEST 1: FORWARD_PASS
    solver = Solver(12, 4, 10)
    DATA = False
    forward_pass(solver)
    forward_pass_1(solver, 2)
    forward_pass_2(solver, 2)

    # TEST 2: BACKWARD_PASS
    solver = Solver(12, 4, 10)
    backward_pass(solver)

    # TEST 3: UPDATE_PRIMAL
    solver = Solver(12, 4, 10)
    update_primal(solver)

    # TEST 4: UPDATE_SLACK
    solver = Solver(12, 4, 10)
    update_slack(solver)

    # TEST 5: UPDATE_DUAL
    solver = Solver(12, 4, 10)
    update_dual(solver)

    # TEST 6: UPDATE_LINEAR_COST
    solver = Solver(12, 4, 10)
    forward_pass(solver)


if __name__ == "__main__":
    run_tests()
