// Quadrotor hovering example
// Make sure in glob_opts.hpp:
// - NSTATES = 12, NINPUTS=4
// - NHORIZON = anything you want
// - tinytype = float if you want to run on microcontrollers
// States: x (m), y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi
// Inputs: u1, u2, u3, u4 (motor thrust 0-1, order from Crazyflie)

// phi, theta, psi are NOT Euler angles, they are Rodiguez parameters
// check this paper for more details: https://roboticexplorationlab.org/papers/planning_with_attitude.pdf

#include <stdio.h>
#include <stdint.h>

#include <tinympc/admm.hpp>
#include <tinympc/admm_rvv.hpp>
#include "problem_data/quadrotor_20hz_params.hpp"
#include "trajectory_data/quadrotor_20hz_y_axis_line.hpp"

#define MSTATUS_VS          0x00000600
#define MSTATUS_FS          0x00006000
#define MSTATUS_XS          0x00018000

#define NUM_PERF_TESTS 10

extern "C"
{

TinyCache cache;
TinyWorkspace work;
TinySettings settings;
TinySolver solver{&settings, &cache, &work};

static inline void enable_vector_operations() {
    unsigned long mstatus;
    
    // Read current mstatus
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    
    // Set VS field to Dirty (11)
    mstatus |= MSTATUS_VS | MSTATUS_FS | MSTATUS_XS;
    
    // Write back updated mstatus
    asm volatile("csrw mstatus, %0" :: "r"(mstatus));
}

static uint64_t read_cycles() {
    uint64_t cycles;
    // asm volatile ("rdcycle %0" : "=r" (cycles));
    asm volatile ("csrr %0, cycle" : "=r" (cycles));
    return cycles;
}


int main()
{
    // General state temporary variables
    printf("Entered main!\n");

    float a[1000];
    float b[1000];
    float c[1000];

    float * a_ptr = a;
    float * b_ptr = b;
    float * c_ptr = c;

    enable_vector_operations();
    uint64_t start, end;

    tiny_VectorNx v1, v2;

    uint64_t total;
    int i;

    // Testing forward_pass_1
    total = 0;
    for (i = 0; i < NUM_PERF_TESTS; i++) {
        start = read_cycles();
        matadd(&a_ptr, &b_ptr, &c_ptr, 100, 10);
        end = read_cycles();
        total += end - start;
    }
    printf("matadd: %lu\n", total / NUM_PERF_TESTS);



}

} /* extern "C" */