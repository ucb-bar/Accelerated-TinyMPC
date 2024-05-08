//
// Created by widyadewi on 2/23/24.
//

#pragma once
#ifndef TINYMPC_ADMM_RVV_HPP
#define TINYMPC_ADMM_RVV_HPP

#include "types_rvv.hpp"
#include "Gemmini/gemmini.h"

// #ifndef USE_MATVEC
// #define USE_MATVEC 1
// #endif

extern "C" {

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

#ifdef USE_GEMMINI
void mvin_matrix(tinytype * data, spad_ptr_t spad_addr, int rows, int cols);
void mvin_vector(tinytype * data, spad_ptr_t spad_addr, int size);
#endif

// u1 = x[:, i] * Kinf; u2 = u1 + d; u[:, i] = -u2
inline void forward_pass_1(TinySolver *solver, int i) {
#ifdef USE_MATVEC
    matvec(solver->cache->Kinf.data, solver->work->x.col(i), solver->work->u1.data, NINPUTS, NSTATES);
#else
    
#ifdef USE_GEMMINI
    // gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
    // printf("Forward pass 1: %d\n", i);
    gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
    // gemmini_config_st(4);
    gemmini_extended_config_st(4, 0, 1.0);
    // TODO REMOVE MVIN
    gemmini_extended3_config_ld(4, 1.0, false, 1);
    // Print x.col(i)
    // printf("x.col(i): ");
    // for(int j = 0; j < NSTATES; j++) {
    //     solver->work->x.col(i)[j] = j + 1.0;
    //     printf("%0.5f ",solver->work->x.col(i)[j]);
    // }
    // printf("\n");

    // gemmini_extended_mvin2(solver->work->x.col(i), x_spad + i*NSTATES, 1, DIM);
    // gemmini_extended_mvin2(solver->work->x.col(i) + DIM, x_spad + i*NSTATES+ DIM, 1, DIM);
    // gemmini_extended_mvin2(solver->work->x.col(i) + 2*DIM, x_spad + i*NSTATES+ 2*DIM, 1, DIM);
    // gemmini_extended_preload(GARBAGE_ADDR, u_spad + i*NINPUTS, 1, 4, 1, 4);
    // gemmini_extended_preload(GARBAGE_ADDR, u_spad + i*NINPUTS, 1, 4, 1, 4);
    // gemmini_extended_preload(GARBAGE_ADDR, u_spad + i*NINPUTS, 4, 4, 4, 4);
    gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
    // gemmini_preload_zeros(u_spad + i*NINPUTS);

    gemmini_compute_preloaded(Kinf_spad, x_spad + i*NSTATES);
    gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
    gemmini_compute_accumulated(Kinf_spad+DIM, x_spad+i*NSTATES+DIM);
    gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
    // gemmini_extended_preload(GARBAGE_ADDR, u_spad + i*NINPUTS, 1, 4, 1, 4);
    gemmini_compute_accumulated(Kinf_spad+2*DIM, x_spad+i*NSTATES+2*DIM);
    gemmini_extended_mvout(solver->work->u1.data, u_spad + i*NINPUTS, 1, 4);
    // gemmini_extended_mvout(solver->work->u1.data, u_spad + i*NINPUTS, 1, 1);
    // gemmini_extended_mvout(solver->work->u1.data+1, u_spad + i*NINPUTS + 1, 1, 1);
    // gemmini_extended_mvout(solver->work->u1.data+2, u_spad + i*NINPUTS + 2, 1, 1);
    // gemmini_extended_mvout(solver->work->u1.data+3, u_spad + i*NINPUTS + 3, 1, 1);
    // gemmini_fence();
    // for(int i = 0; i < NINPUTS; i++) {
    //     printf("u1[%d]: %0.5f\n", i, solver->work->u1.data[i]);
    // }

    // matmul(solver->work->x.col(i), solver->cache->Kinf.data, solver->work->u1.data, 1, NINPUTS, NSTATES);
    // for(int i = 0; i < NINPUTS; i++) {
    //     printf("u1[%d]: %0.5f\n", i, solver->work->u1.data[i]);
    // }
    // exit(0);
#else
    matmul(solver->work->x.col(i), solver->cache->Kinf.data, solver->work->u1.data, 1, NINPUTS, NSTATES);
#endif
#endif
#ifdef USE_GEMMINI

    // TODO REMOVE MVIN
    // gemmini_extended_mvout(solver->work->u1.data, u_spad + i*NINPUTS, 1, 4);
    // gemmini_fence();
    gemmini_extended3_config_ld(4, 1.0, false, 1);
    gemmini_extended_mvin2(solver->work->d.col(i), d_spad + i*NSTATES, 1, DIM);
    gemmini_extended_mvin2(solver->work->d.col(i)+DIM, d_spad + i*NSTATES+DIM, 1, DIM);
    gemmini_extended_mvin2(solver->work->d.col(i)+2*DIM, d_spad + i*NSTATES+2*DIM, 1, DIM);
    // printf("Step 1!\n");
    gemmini_extended_preload(GARBAGE_ADDR, u_spad + i*NINPUTS, 1, 4, 1, 4);
    gemmini_compute_accumulated(I_spad, d_spad + i*NSTATES);
    // printf("Step 2!\n");
    gemmini_extended_preload(GARBAGE_ADDR, u_spad + i*NINPUTS, 1, 4, 1, 4);
    gemmini_compute_preloaded(nI_spad, u_spad + i*NINPUTS);
    gemmini_extended_mvout(solver->work->u.col(i), u_spad + i*NINPUTS, 1, 4);
    // gemmini_extended_preload(GARBAGE_ADDR, temp_spad, 1, 4, 1, 4);
    // for(int i = 0; i < NINPUTS; i++) {
    //     printf("u1[%d]: %0.5f\n", i, solver->work->u1.data[i]);
    // }
    // exit(0);
    // printf("Step 0!: %p\n", solver->work->d.col(i));
    // gemmini_extended_mvout(solver->work->u1.data, 0, 1, 4);
    // gemmini_extended_mvout(solver->work->d.col(i), d_spad + i*NINPUTS, 1, 4);
    // gemmini_extended_mvout(solver->work->d.col(i) + 1, d_spad + i*NINPUTS + 1, 1, 1);
    // gemmini_extended_mvout(solver->work->d.col(i) + 3, d_spad + i*NINPUTS + 3, 1, 1);
    // gemmini_fence();
    // gemmini_extended_mvout(solver->work->d.col(i), d_spad + i*NINPUTS, 1, 1);
    // gemmini_fence();
    // gemmini_extended_mvout(solver->work->d.col(i) + 2, d_spad + i*NINPUTS + 2, 1, 1);
    // gemmini_fence();
    // for(int j = 0; j < NINPUTS; j++) {
    //     printf("uf1[%d]: %0.5f\n", j, solver->work->u.col(i)[j]);
    // }

    // TRACE_CHECKSUM(forward_pass_1, solver->work->u1);
    // matneg(solver->work->u2.data, solver->work->u.col(i), 1, NINPUTS);
    // TRACE_CHECKSUM(forward_pass_1, solver->work->u);
    // matadd(solver->work->u1.data, solver->work->d.col(i), solver->work->u2.data, 1, NINPUTS);
    // matneg(solver->work->u2.data, solver->work->u.col(i), 1, NINPUTS);
    // for(int j = 0; j < NINPUTS; j++) {
    //     printf("uf1[%d]: %0.5f\n", j, solver->work->u.col(i)[j]);
    // }
    // exit(0);
#else
    TRACE_CHECKSUM(forward_pass_1, solver->work->u1);
    matadd(solver->work->u1.data, solver->work->d.col(i), solver->work->u2.data, 1, NINPUTS);
    TRACE_CHECKSUM(forward_pass_1, solver->work->u2);
    matneg(solver->work->u2.data, solver->work->u.col(i), 1, NINPUTS);
    TRACE_CHECKSUM(forward_pass_1, solver->work->u);
#endif
}

// x[:, i+1] = Adyn * x[:, i] + Bdyn * u[:, i]
inline void forward_pass_2(TinySolver *solver, int i) {
#ifdef USE_MATVEC
    matvec(solver->work->Adyn.data, solver->work->x.col(i), solver->work->x1.data, NSTATES, NSTATES);
    matvec(solver->work->Bdyn.data, solver->work->u.col(i), solver->work->x2.data, NSTATES, NINPUTS);
    matadd(solver->work->x1.data, solver->work->x2.data, solver->work->x.col(i + 1), 1, NSTATES);
#else
#ifdef USE_GEMMINI
    // printf("Forward pass 2: %d\n", i);
    // printf("Step 4!\n");
    gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
    gemmini_config_st(4);
    gemmini_extended3_config_ld(4, 1.0, false, 1);
    // gemmini_extended_mvin2(solver->work->x.col(i), x_spad + i*NSTATES, 1, NSTATES);
    // gemmini_extended_mvin2(solver->work->u.col(i), u_spad + i*NINPUTS, 1, NINPUTS);

    // printf("Step 5!\n");
    for(size_t j = 0; j < NSTATES/DIM; j++){
        gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
        gemmini_compute_preloaded  (Adyn_spad + 0*NSTATES + j*DIM,         x_spad + i*NSTATES);
        gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
        gemmini_compute_accumulated(Adyn_spad + 1*NSTATES + j*DIM,   x_spad + i*NSTATES + 1*DIM);
        gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
        gemmini_compute_accumulated(Adyn_spad + 2*NSTATES + j*DIM, x_spad + i*NSTATES + 2*DIM);

        gemmini_extended_preload(GARBAGE_ADDR, x_spad + (i+1)*NSTATES + j*DIM, 1, 4, 1, 4);
        gemmini_compute_accumulated(Bdyn_spad + j*DIM, u_spad + i*NINPUTS);
        gemmini_extended_mvout(solver->work->x.col(i+1) + j*DIM, x_spad + (i+1)*NSTATES + j*DIM, 1, 4);
    }
    // printf("Step 6!\n");
    // gemmini_fence();
    // printf("xf1:  ");
    // for(int j = 0; j < NSTATES; j++) {
    //     printf("%0.5f ", solver->work->x.col(i+1)[j]);
    // }
    // printf("\n");

    // printf("Step 7!\n");
    // exit(0);

    // matmul(solver->work->x.col(i), solver->work->Adyn.data, solver->work->x1.data, 1, NSTATES, NSTATES);
    // matmul(solver->work->u.col(i), solver->work->Bdyn.data, solver->work->x2.data, 1, NSTATES, NINPUTS);
    // matadd(solver->work->x1.data, solver->work->x2.data, solver->work->x.col(i + 1), 1, NSTATES);
    // printf("xf1:  ");
    // for(int j = 0; j < NSTATES; j++) {
    //     printf("%0.5f ", solver->work->x.col(i+1)[j]);
    // }
    // printf("\n");
    // exit(0);
#else
    matmul(solver->work->x.col(i), solver->work->Adyn.data, solver->work->x1.data, 1, NSTATES, NSTATES);
    matmul(solver->work->u.col(i), solver->work->Bdyn.data, solver->work->x2.data, 1, NSTATES, NINPUTS);
    matadd(solver->work->x1.data, solver->work->x2.data, solver->work->x.col(i + 1), 1, NSTATES);
#endif
#endif
    TRACE_CHECKSUM(forward_pass_2, solver->work->x);
}

// d[:, i] = Quu_inv * (BdynT * p[:, i+1] + r[:, i]);
inline void backward_pass_1(TinySolver *solver, int i) {
#ifdef USE_MATVEC
    matvec(solver->work->BdynT.data, solver->work->p.col(i + 1), solver->work->u1.data, NINPUTS, NSTATES);
    matadd(solver->work->r.col(i), solver->work->u1.data, solver->work->u2.data, 1, NINPUTS);
    matvec(solver->cache->Quu_inv.data, solver->work->u2.data, solver->work->d.col(i), NINPUTS, NINPUTS);
#else
#ifdef USE_GEMMINI
    // printf("Backward pass 1: %d\n", i);
    gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
    gemmini_config_st(4);
    gemmini_extended3_config_ld(4, 1.0, false, 1);
    // gemmini_extended_mvin2(solver->work->p.col(i+1), p_spad + (i+1)*NSTATES, 1, DIM);
    // gemmini_extended_mvin2(solver->work->p.col(i+1) + DIM, p_spad + (i+1)*NSTATES + DIM, 1, DIM);
    // gemmini_extended_mvin2(solver->work->p.col(i+1) + 2*DIM, p_spad + (i+1)*NSTATES + 2*DIM, 1, DIM);
    gemmini_extended_mvin2(solver->work->r.col(i), r_spad + i*NINPUTS, 1, NINPUTS);

    // printf("p1:  ");
    // for(int j = 0; j < NSTATES; j++) {
    //     printf("%0.5f ", solver->work->p.col(i+1)[j]);
    // }
    // printf("\n");

    // printf("r1:  ");
    // for(int j = 0; j < NINPUTS; j++) {
    //     printf("%0.5f ", solver->work->r.col(i)[j]);
    // }
    // printf("\n");

    gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
    gemmini_compute_preloaded(BdynT_spad, p_spad + (i+1)*NSTATES);
    gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
    gemmini_compute_accumulated(BdynT_spad+DIM, p_spad+(i+1)*NSTATES+DIM);
    // gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
    // gemmini_extended_preload(GARBAGE_ADDR, d_spad + i*NINPUTS, 1, 4, 1, 4);
    gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
    gemmini_compute_accumulated(BdynT_spad+2*DIM, p_spad+(i+1)*NSTATES+2*DIM);

    // gemmini_extended_mvout(solver->work->u1.data, d_spad + i*NINPUTS, 1, 4);
    // gemmini_fence();
    // printf("u1:  ");
    // for(int j = 0; j < NINPUTS; j++) {
    //     printf("%0.5f ", solver->work->u1.data[j]);
    // }
    // printf("\n");


    gemmini_extended_preload(GARBAGE_ADDR, d_spad + i*NINPUTS, 1, 4, 1, 4);
    gemmini_compute_accumulated(I_spad, r_spad+i*NINPUTS);

    gemmini_extended_mvout(solver->work->u2.data, d_spad + i*NINPUTS, 1, 4);
    // gemmini_fence();
    // printf("u2:  ");
    // for(int j = 0; j < NINPUTS; j++) {
    //     printf("%0.5f ", solver->work->u2.data[j]);
    // }
    // printf("\n");
    gemmini_extended_preload(GARBAGE_ADDR, d_spad + i*NINPUTS, 1, 4, 1, 4);
    gemmini_compute_preloaded(Quu_inv_spad, d_spad+i*NINPUTS);

    gemmini_extended_mvout(solver->work->d.col(i), d_spad + i*NINPUTS, 1, 4);
    // gemmini_fence();

    // printf("d1:  ");
    // for(int j = 0; j < NINPUTS; j++) {
    //     printf("%0.5f ", solver->work->d.col(i)[j]);
    // }
    // printf("\n");

    // matmul(solver->work->p.col(i + 1), solver->work->BdynT.data, solver->work->u1.data, 1, NINPUTS, NSTATES);
    // matadd(solver->work->r.col(i), solver->work->u1.data, solver->work->u2.data, 1, NINPUTS);
    // matmul(solver->work->u2.data, solver->cache->Quu_inv.data, solver->work->d.col(i), 1, NINPUTS, NINPUTS);
    // printf("u1:  ");
    // for(int j = 0; j < NINPUTS; j++) {
    //     printf("%0.5f ", solver->work->u1.data[j]);
    // }
    // printf("\n");
    // printf("u2:  ");
    // for(int j = 0; j < NINPUTS; j++) {
    //     printf("%0.5f ", solver->work->u2.data[j]);
    // }
    // printf("\n");
    // printf("d2:  ");
    // for(int j = 0; j < NINPUTS; j++) {
    //     printf("%0.5f ", solver->work->d.col(i)[j]);
    // }
    // printf("\n");
    // exit(0);
#else
    matmul(solver->work->p.col(i + 1), solver->work->BdynT.data, solver->work->u1.data, 1, NINPUTS, NSTATES);
    matadd(solver->work->r.col(i), solver->work->u1.data, solver->work->u2.data, 1, NINPUTS);
    matmul(solver->work->u2.data, solver->cache->Quu_inv.data, solver->work->d.col(i), 1, NINPUTS, NINPUTS);

#endif
#endif
    TRACE_CHECKSUM(backward_pass_1, solver->work->d);
}

// p[:, i] = q[:, i] + AmBKt * p[:, i + 1] - KinfT * r[:, i]
inline void backward_pass_2(TinySolver *solver, int i) {
#ifdef USE_MATVEC
    matvec(solver->cache->AmBKt.data, solver->work->p.col(i + 1), solver->work->x1.data, NSTATES, NSTATES);
    matvec(solver->cache->KinfT.data, solver->work->r.col(i), solver->work->x2.data, NSTATES, NINPUTS);
    matsub(solver->work->x1.data, solver->work->x2.data, solver->work->x3.data, 1, NSTATES);
    matadd(solver->work->x3.data, solver->work->q.col(i), solver->work->p.col(i), 1, NSTATES);
    TRACE_CHECKSUM(backward_pass_2, solver->work->p);
#else
    // printf("Backward pass 2: %d\n", i);
    #ifdef USE_GEMMINI
    gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
    gemmini_config_st(4);
    gemmini_extended3_config_ld(4, 1.0, false, 1);
    // gemmini_extended_mvin2(solver->work->p.col(i+1), p_spad + (i+1)*NSTATES, 1, DIM);
    // gemmini_extended_mvin2(solver->work->p.col(i+1) + DIM, p_spad + (i+1)*NSTATES + DIM, 1, DIM);
    // gemmini_extended_mvin2(solver->work->p.col(i+1) + 2*DIM, p_spad + (i+1)*NSTATES + 2*DIM, 1, DIM);
    
    gemmini_extended_mvin2(solver->work->q.col(i), q_spad + i*NSTATES, 1, DIM);
    gemmini_extended_mvin2(solver->work->q.col(i) +DIM, q_spad + i*NSTATES + DIM, 1, DIM);
    gemmini_extended_mvin2(solver->work->q.col(i) +2*DIM, q_spad + i*NSTATES + 2*DIM, 1, DIM);

    gemmini_extended_mvin2(solver->work->r.col(i), r_spad + i*NINPUTS, 1, NINPUTS);

    for(size_t j = 0; j < NSTATES/DIM; j++){
        gemmini_extended_preload(GARBAGE_ADDR, p_spad + i*NSTATES + j*DIM, 1, 4, 1, 4);
        gemmini_compute_preloaded(KinfT_spad + j*DIM, r_spad + i*NINPUTS);
        gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
        gemmini_compute_preloaded(nI_spad, p_spad + i*NSTATES + j*DIM);
        // gemmini_extended_mvout(solver->work->x2.data + j*DIM, p_spad + i*NSTATES + j*DIM, 1, 4);

        gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
        gemmini_compute_accumulated(AmBKt_spad + 0*NSTATES + j*DIM,   p_spad + (i+1)*NSTATES);
        gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
        gemmini_compute_accumulated(AmBKt_spad + 1*NSTATES + j*DIM,   p_spad + (i+1)*NSTATES + 1*DIM);
        gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
        gemmini_compute_accumulated(AmBKt_spad + 2*NSTATES + j*DIM,   p_spad + (i+1)*NSTATES + 2*DIM);
        // gemmini_extended_mvout(solver->work->x3.data + j*DIM, p_spad + i*NSTATES + j*DIM, 1, 4);

        gemmini_extended_preload(GARBAGE_ADDR, p_spad + (i)*NSTATES + j*DIM, 1, 4, 1, 4);
        gemmini_compute_accumulated(I_spad, q_spad + i*NSTATES + j*DIM);
        gemmini_extended_mvout(solver->work->p.col(i) + j*DIM, p_spad + i*NSTATES + j*DIM, 1, 4);

    }
    // gemmini_fence();
    // printf("p1:  ");
    // for(int j = 0; j < NSTATES; j++) {
    //     printf("%0.5f ", solver->work->p.col(i)[j]);
    // }
    // printf("\n");


    // matmul(solver->work->p.col(i + 1), solver->cache->AmBKt.data, solver->work->x1.data, 1, NSTATES, NSTATES);
    // matmul(solver->work->r.col(i), solver->cache->KinfT.data, solver->work->x2.data, 1, NSTATES, NINPUTS);
    // matsub(solver->work->x1.data, solver->work->x2.data, solver->work->x3.data, 1, NSTATES);
    // matadd(solver->work->x3.data, solver->work->q.col(i), solver->work->p.col(i), 1, NSTATES);

    // printf("p1:  ");
    // for(int j = 0; j < NSTATES; j++) {
    //     printf("%0.5f ", solver->work->p.col(i)[j]);
    // }
    // printf("\n");
    // exit(0);

    #else
    matmul(solver->work->p.col(i + 1), solver->cache->AmBKt.data, solver->work->x1.data, 1, NSTATES, NSTATES);
    matmul(solver->work->r.col(i), solver->cache->KinfT.data, solver->work->x2.data, 1, NSTATES, NINPUTS);
    matsub(solver->work->x1.data, solver->work->x2.data, solver->work->x3.data, 1, NSTATES);
    matadd(solver->work->x3.data, solver->work->q.col(i), solver->work->p.col(i), 1, NSTATES);
    TRACE_CHECKSUM(backward_pass_2, solver->work->p);
    #endif
#endif
}

// y u znew  g x vnew
inline void update_dual_1(TinySolver *solver) {
    #ifdef USE_GEMMINI
    /*gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
    gemmini_config_st(4);
    gemmini_extended3_config_ld(4, 1.0, false, 1);
    for(size_t i = 0; i < NHORIZON-1; i++) {
        gemmini_extended_mvin2(solver->work->y.col(i), y_spad + i*NSTATES, 1, DIM);
        gemmini_extended_mvin2(solver->work->y.col(i) +DIM, y_spad + i*NSTATES + DIM, 1, DIM);
        gemmini_extended_mvin2(solver->work->y.col(i) +2*DIM, y_spad + i*NSTATES + 2*DIM, 1, DIM);

        gemmini_extended_mvin2(solver->work->g.col(i), g_spad + i*NSTATES, 1, DIM);
        gemmini_extended_mvin2(solver->work->g.col(i) +DIM, g_spad + i*NSTATES + DIM, 1, DIM);
        gemmini_extended_mvin2(solver->work->g.col(i) +2*DIM, g_spad + i*NSTATES + 2*DIM, 1, DIM);

        // gemmini_extended_mvin2(solver->work->u.col(i), u_spad + i*NINPUTS, 1, DIM);
        gemmini_extended_mvin2(solver->work->znew.col(i), znew_spad + i*NINPUTS, 1, DIM);

        gemmini_extended_mvin2(solver->work->vnew.col(i), vnew_spad + i*NSTATES, 1, DIM);
        gemmini_extended_mvin2(solver->work->vnew.col(i) +DIM, vnew_spad + i*NSTATES + DIM, 1, DIM);
        gemmini_extended_mvin2(solver->work->vnew.col(i) +2*DIM, vnew_spad + i*NSTATES + 2*DIM, 1, DIM);

        gemmini_extended_preload(u_spad + i* NINPUTS, GARBAGE_ADDR, 1, 4, 1, 4);
        gemmini_compute_preloaded(I_spad,   y_spad + i*NINPUTS);
        gemmini_extended_preload(GARBAGE_ADDR, y_spad + i*NINPUTS, 1, 4, 1, 4);
        gemmini_compute_accumulated(nI_spad,   znew_spad + i*NINPUTS);

        gemmini_extended_mvout(solver->work->y.col(i), y_spad + i*NSTATES, 1, 4);

        for(size_t j = 0; j < 3; j++ ) {
            gemmini_extended_preload(x_spad + i *NSTATES + j*DIM, GARBAGE_ADDR, 1, 4, 1, 4);
            gemmini_compute_preloaded(I_spad,   g_spad + i*NSTATES + j*DIM);
            gemmini_extended_preload(GARBAGE_ADDR, g_spad + i*NSTATES + j*DIM, 1, 4, 1, 4);
            gemmini_compute_accumulated(nI_spad,  vnew_spad + i*NSTATES + j*DIM);

            gemmini_extended_mvout(solver->work->g.col(i) + j*DIM, g_spad + i*NSTATES + j*DIM, 1, 4);
        }
    }
    gemmini_fence(); */

    // gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, false, false);
    // gemmini_extended3_config_ld(sizeof(float) * DIM, 1.0, false, 0);
    // gemmini_extended3_config_ld(sizeof(float) * DIM, -1.0, false, 2);
    // gemmini_config_st(sizeof(float) * DIM);

    // uint32_t sp_base =  3 << (ADDR_LEN-2);
    // uint32_t sp_base_clear = 1 << (ADDR_LEN-1);

    // uint8_t remainder;

    // sp_base =  3 << (ADDR_LEN-2);
    // sp_base_clear = 1 << (ADDR_LEN-1);
    // gemmini_fence();

    // size_t i = 0;
    // // for(i = 0; i < (NHORIZON-1) / DIM; i++) {
    // //     gemmini_extended_mvin(solver->work->y.data + i*DIM*DIM, sp_base_clear + i*DIM, DIM, DIM);
    // //     gemmini_extended_mvin(solver->work->u.data + i*DIM*DIM, sp_base + i*DIM, DIM, DIM);
    // //     gemmini_extended_mvin3(solver->work->znew.data + i*DIM*DIM, sp_base + i*DIM, DIM, DIM);
    // //     // gemmini_extended_mvout(solver->work->y.data + i*DIM*DIM, sp_base + i*DIM, DIM, DIM);
    // // }
    // // remainder = (NHORIZON-1) % DIM;
    // // // printf("i: %d\n", i);
    // // // printf("remainder: %d\n", remainder);
    // // gemmini_extended_mvin(solver->work->y.data + i*DIM*DIM, sp_base_clear + i*DIM, DIM, remainder);
    // // gemmini_extended_mvin(solver->work->u.data + i*DIM*DIM, sp_base + i*DIM, DIM, remainder);
    // // gemmini_extended_mvin3(solver->work->znew.data + i*DIM*DIM, sp_base + i*DIM, DIM, remainder);
    // // gemmini_extended_mvout(solver->work->y.data + i*DIM*DIM, sp_base + i*DIM, DIM, remainder);

    // sp_base =  (3 << (ADDR_LEN-2)) + 32;
    // sp_base_clear = (1 << (ADDR_LEN-1)) + 32;

    // for(i = 0; i < (NHORIZON * NSTATES) / (DIM * DIM); i++) {
    //     gemmini_extended_mvin(solver->work->g.data + i*DIM*DIM, sp_base_clear + i*DIM, DIM, DIM);
    //     gemmini_extended_mvin(solver->work->x.data + i*DIM*DIM, sp_base + i*DIM, DIM, DIM);
    //     gemmini_extended_mvin3(solver->work->vnew.data + i*DIM*DIM, sp_base + i*DIM, DIM, DIM);
    //     // gemmini_extended_mvout(solver->work->g.data + i*DIM*DIM, sp_base + i*DIM, DIM, DIM);
    // }
    // remainder = ((NHORIZON * NSTATES) / DIM) % DIM;
    // // printf("i: %d\n", i);
    // // printf("remainder: %d\n", remainder);
    // gemmini_extended_mvin(solver->work->g.data + i*DIM*DIM, sp_base_clear + i*DIM, DIM, remainder);
    // gemmini_extended_mvin(solver->work->x.data + i*DIM*DIM, sp_base + i*DIM, DIM, remainder);
    // gemmini_extended_mvin3(solver->work->vnew.data + i*DIM*DIM, sp_base + i*DIM, DIM, remainder);
    // // gemmini_extended_mvout(solver->work->g.data + i*DIM*DIM, sp_base + i*DIM, DIM, remainder);

    // gemmini_fence();

    // sp_base =  3 << (ADDR_LEN-2);
    // sp_base_clear = 1 << (ADDR_LEN-1);

    // // i = 0;
    // // for(i = 0; i < (NHORIZON-1) / DIM; i++) {
    // //     gemmini_extended_mvout(solver->work->m1.data + i*DIM*DIM, sp_base + i*DIM, DIM, DIM);
    // // }
    // // remainder = (NHORIZON-1) % DIM;
    // // // printf("i: %d\n", i);
    // // // printf("remainder: %d\n", remainder);
    // // gemmini_extended_mvout(solver->work->m1.data + i*DIM*DIM, sp_base + i*DIM, DIM, remainder);

    // sp_base =  (3 << (ADDR_LEN-2)) + 32;
    // sp_base_clear = (1 << (ADDR_LEN-1)) + 32;

    // for(i = 0; i < (NHORIZON * NSTATES) / (DIM * DIM); i++) {
    //     gemmini_extended_mvout(solver->work->g.data + i*DIM*DIM, sp_base + i*DIM, DIM, DIM);
    // }
    // remainder = ((NHORIZON * NSTATES) / DIM) % DIM;
    // gemmini_extended_mvout(solver->work->g.data + i*DIM*DIM, sp_base + i*DIM, DIM, remainder);
    // gemmini_fence();

    // // printf("RTL Check\n");
    // // // printf("y:  ");
    // // // for(int j = 0; j < (NHORIZON-1) * NINPUTS; j++) {
    // // //     printf("%0.5f ", solver->work->m1.data[j]);
    // // // }
    // // // printf("\n");
    // // printf("\n");
    // // printf("g:  ");
    // // for(int j = 0; j < (NHORIZON) * NSTATES; j++) {
    // //     printf("%0.5f ", solver->work->s1.data[j]);
    // // }
    // // printf("\n");
    // // printf("\n");

    gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
    gemmini_extended_config_st(DIM*sizeof(float), 0, 1.0);
    gemmini_extended3_config_ld(sizeof(float) * DIM, 1.0, false, 0);

    // gemmini_extended_preload(GARBAGE_ADDR, nrI_spad, 4, 4, 4, 4);
    // gemmini_compute_preloaded(rI_spad,   I_spad);
    // gemmini_extended_mvout(nrI, nrI_spad, 4, 4);
    // gemmini_fence();
    size_t i;
    uint8_t remainder;

    remainder = DIM;
    for(i = 0; i < (NHORIZON * NSTATES) / (DIM * DIM); i++) {
        gemmini_extended_mvin(solver->work->g.data + i*DIM*DIM, temp_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->x.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
        gemmini_extended_mvin(solver->work->vnew.data + i*DIM*DIM, temp_spad + 2*DIM, DIM, remainder);
        gemmini_extended_preload(temp_spad, GARBAGE_ADDR, 4, 4, 4, 4); // preload g
        gemmini_compute_preloaded(I_spad,   temp_spad + 1*DIM); // x
        gemmini_extended_preload(GARBAGE_ADDR, temp_spad+ 3*DIM, 4, 4, 4, 4); 
        gemmini_compute_accumulated(nI_spad,   temp_spad + 2*DIM); // vnew
        gemmini_extended_mvout(solver->work->g.data + i*DIM*DIM, temp_spad+ 3*DIM, DIM, remainder);
    }
    remainder = ((NHORIZON * NSTATES) / DIM) % DIM;
    gemmini_extended_mvin(solver->work->g.data + i*DIM*DIM, temp_spad, DIM, remainder);
    gemmini_extended_mvin(solver->work->x.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
    gemmini_extended_mvin(solver->work->vnew.data + i*DIM*DIM, temp_spad + 2*DIM, DIM, remainder);
    gemmini_extended_preload(temp_spad, GARBAGE_ADDR, 4, 4, 4, 4); // preload g
    gemmini_compute_preloaded(I_spad,   temp_spad + 1*DIM); // x
    gemmini_extended_preload(GARBAGE_ADDR, temp_spad+ 3*DIM, 4, 4, 4, 4); 
    gemmini_compute_accumulated(nI_spad,   temp_spad + 2*DIM); // vnew
    gemmini_extended_mvout(solver->work->g.data + i*DIM*DIM, temp_spad+ 3*DIM, DIM, remainder);

    remainder = DIM;
    for(i = 0; i < ((NHORIZON-1) * NINPUTS) / (DIM * DIM); i++) {
        gemmini_extended_mvin(solver->work->y.data + i*DIM*DIM, temp_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->u.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
        gemmini_extended_mvin(solver->work->znew.data + i*DIM*DIM, temp_spad + 2*DIM, DIM, remainder);
        gemmini_extended_preload(temp_spad, GARBAGE_ADDR, 4, 4, 4, 4); // preload y
        gemmini_compute_preloaded(I_spad,   temp_spad + 1*DIM); // u
        gemmini_extended_preload(GARBAGE_ADDR, temp_spad+ 3*DIM, 4, 4, 4, 4); 
        gemmini_compute_accumulated(nI_spad,   temp_spad + 2*DIM); // znew
        gemmini_extended_mvout(solver->work->y.data + i*DIM*DIM, temp_spad+ 3*DIM, DIM, remainder);
    }
    remainder = (((NHORIZON-1) * NINPUTS) / DIM) % DIM;
    gemmini_extended_mvin(solver->work->y.data + i*DIM*DIM, temp_spad, DIM, remainder);
    gemmini_extended_mvin(solver->work->u.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
    gemmini_extended_mvin(solver->work->znew.data + i*DIM*DIM, temp_spad + 2*DIM, DIM, remainder);
    gemmini_extended_preload(temp_spad, GARBAGE_ADDR, 4, 4, 4, 4); // preload y
    gemmini_compute_preloaded(I_spad,   temp_spad + 1*DIM); // u
    gemmini_extended_preload(GARBAGE_ADDR, temp_spad+ 3*DIM, 4, 4, 4, 4); 
    gemmini_compute_accumulated(nI_spad,   temp_spad + 2*DIM); // znew
    gemmini_extended_mvout(solver->work->y.data + i*DIM*DIM, temp_spad+ 3*DIM, DIM, remainder);

    gemmini_fence();


    // matadd(solver->work->y.data, solver->work->u.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    // matsub(solver->work->m1.data, solver->work->znew.data, solver->work->y.data, NHORIZON - 1, NINPUTS);
    // matadd(solver->work->g.data, solver->work->x.data, solver->work->s1.data, NHORIZON, NSTATES);
    // matsub(solver->work->s1.data, solver->work->vnew.data, solver->work->g.data, NHORIZON, NSTATES);

    // printf("SW Check\n");
    // // printf("y:  ");
    // // for(int j = 0; j < (NHORIZON-1) * NINPUTS; j++) {
    // //     printf("%0.5f ", solver->work->y.data[j]);
    // // }
    // // printf("\n");
    // printf("\n");
    // printf("g:  ");
    // for(int j = 0; j < (NHORIZON) * NSTATES; j++) {
    //     printf("%0.5f ", solver->work->g.data[j]);
    // }
    // printf("\n");
    // exit(0);

    #else
    matadd(solver->work->y.data, solver->work->u.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    matsub(solver->work->m1.data, solver->work->znew.data, solver->work->y.data, NHORIZON - 1, NINPUTS);
    matadd(solver->work->g.data, solver->work->x.data, solver->work->s1.data, NHORIZON, NSTATES);
    matsub(solver->work->s1.data, solver->work->vnew.data, solver->work->g.data, NHORIZON, NSTATES);
    TRACE_CHECKSUM(update_dual_1, solver->work->g);
    #endif
}

// Box constraints on input
inline void update_slack_1(TinySolver *solver) {
    #ifdef USE_GEMMINI
    gemmini_extended_config_ex(OUTPUT_STATIONARY, RELU, 0, 1, false, false);
    gemmini_extended_config_st(DIM*sizeof(float), 0, 1.0);
    gemmini_extended3_config_ld(sizeof(float) * DIM, 1.0, false, 0);
    gemmini_extended3_config_ld(sizeof(float) * DIM, -1.0, false, 2);


    size_t i;
    uint8_t remainder;
    // TODO HANDLE IF

    const uint32_t temp_y_spad = temp_spad + 0;
    const uint32_t temp_u_spad = temp_spad + 1*DIM;
    const uint32_t temp_u_max_spad = temp_spad + 5*DIM;
    const uint32_t temp_u_min_spad = temp_spad + 3*DIM;

    const uint32_t temp_clip_spad = temp_spad + 2*DIM;

    remainder = DIM;
    for(i = 0; i < ((NHORIZON-1) * NINPUTS) / (DIM * DIM); i++) {
        gemmini_extended_config_ex(OUTPUT_STATIONARY, RELU, 0, 1, false, false);
        gemmini_extended_mvin3(solver->work->y.data + i*DIM*DIM, temp_y_spad, DIM, remainder);
        gemmini_extended_mvin3(solver->work->u.data + i*DIM*DIM, temp_u_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->u_max.data + i*DIM*DIM, temp_u_max_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->u_min.data + i*DIM*DIM, temp_u_min_spad, DIM, remainder);

        gemmini_extended_preload(temp_y_spad, GARBAGE_ADDR, 4, 4, 4, 4); 
        gemmini_compute_preloaded(I_spad,   temp_u_spad);  // add inputs, negated
        gemmini_extended_preload(GARBAGE_ADDR, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_accumulated(I_spad,   temp_u_max_spad);  // add the maximum


        gemmini_extended_preload(temp_u_max_spad, GARBAGE_ADDR, 4, 4, 4, 4); 
        gemmini_compute_preloaded(nI_spad,   temp_u_min_spad);  // subtract the minimum 
        gemmini_extended_preload(GARBAGE_ADDR, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_accumulated(nI_spad,   temp_clip_spad);  // 

        gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
        gemmini_extended_preload(temp_u_min_spad, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_preloaded(I_spad,   temp_clip_spad);  // add inputs, negated

        gemmini_extended_mvout(solver->work->znew.data + i*DIM*DIM, temp_clip_spad, DIM, remainder);

        // exit(0);
    }
    remainder = (((NHORIZON-1) * NINPUTS) / DIM) % DIM;
        gemmini_extended_config_ex(OUTPUT_STATIONARY, RELU, 0, 1, false, false);
        gemmini_extended_mvin3(solver->work->y.data + i*DIM*DIM, temp_y_spad, DIM, remainder);
        gemmini_extended_mvin3(solver->work->u.data + i*DIM*DIM, temp_u_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->u_max.data + i*DIM*DIM, temp_u_max_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->u_min.data + i*DIM*DIM, temp_u_min_spad, DIM, remainder);

        gemmini_extended_preload(temp_y_spad, GARBAGE_ADDR, 4, 4, 4, 4); 
        gemmini_compute_preloaded(I_spad,   temp_u_spad);  // add inputs, negated
        gemmini_extended_preload(GARBAGE_ADDR, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_accumulated(I_spad,   temp_u_max_spad);  // add the maximum


        gemmini_extended_preload(temp_u_max_spad, GARBAGE_ADDR, 4, 4, 4, 4); 
        gemmini_compute_preloaded(nI_spad,   temp_u_min_spad);  // subtract the minimum 
        gemmini_extended_preload(GARBAGE_ADDR, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_accumulated(nI_spad,   temp_clip_spad);  // 

        gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
        gemmini_extended_preload(temp_u_min_spad, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_preloaded(I_spad,   temp_clip_spad);  // add inputs, negated

        gemmini_extended_mvout(solver->work->znew.data + i*DIM*DIM, temp_clip_spad, DIM, remainder);
    gemmini_fence();
    #else
    matadd(solver->work->u.data, solver->work->y.data, solver->work->znew.data, NHORIZON - 1, NINPUTS);
    if (solver->settings->en_input_bound) {
        cwisemax(solver->work->u_min.data, solver->work->znew.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
        cwisemin(solver->work->u_max.data, solver->work->m1.data, solver->work->znew.data, NHORIZON - 1, NINPUTS);
    }
    TRACE_CHECKSUM(update_slack_1, solver->work->znew);
    #endif
}

// Box constraints on state
inline void update_slack_2(TinySolver *solver) {
    #ifdef USE_GEMMINI
    // gemmini_extended3_config_ld(sizeof(float) * DIM, 1.0, false, 0);
    // gemmini_config_st(sizeof(float) * DIM);
    // for(size_t i = 0; i < NHORIZON * (NSTATES/DIM); i+=DIM) {
    //     if(i == 0) {

    //     } else {

    //     }
    // }
    // gemmini_extended3_config_ld(sizeof(float) * DIM, 1.0, false, 0);
    // gemmini_extended3_config_ld(sizeof(float) * DIM, -1.0, false, 2);
    // gemmini_config_st(sizeof(float) * DIM);

    // uint32_t sp_base =  3 << (ADDR_LEN-2);
    // uint32_t sp_base_clear = 1 << (ADDR_LEN-1);

    // size_t i;
    // for(i = 0; i < (NHORIZON * NSTATES) / (DIM * DIM); i++) {
    //     gemmini_extended_mvin(solver->work->x.data + i*DIM*DIM, sp_base_clear + i*DIM, DIM, DIM);
    //     gemmini_extended_mvin(solver->work->g.data + i*DIM*DIM, sp_base + i*DIM, DIM, DIM);
    //     gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM, sp_base + i*DIM, DIM, DIM);
    // }
    // uint8_t remainder = ((NHORIZON * NSTATES) / DIM) % DIM;
    // // printf("i: %d\n", i);
    // // printf("remainder: %d\n", remainder);
    // gemmini_extended_mvin(solver->work->x.data + i*DIM*DIM, sp_base_clear + i*DIM, DIM, remainder);
    // gemmini_extended_mvin(solver->work->g.data + i*DIM*DIM, sp_base + i*DIM, DIM, remainder);
    // gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM, sp_base + i*DIM, DIM, remainder);
    // gemmini_fence();
    // gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
    // // gemmini_extended_config_st(DIM*sizeof(float), 0, 1.0);
    // gemmini_extended2_config_st(DIM*sizeof(float), 0, 1.0, 1, 2, DIM, 1, 1, 2, 4, 0, 1);
    // gemmini_extended3_config_ld(sizeof(float) * DIM, 1.0, false, 0);
    // gemmini_extended3_config_ld(sizeof(float) * DIM, -1.0, false, 2);


    // size_t i;
    // uint8_t remainder;
    // // TODO HANDLE IF

    // remainder = DIM;
    // for(i = 0; i < (NHORIZON * NSTATES) / (DIM * DIM); i++) {
    //     gemmini_extended_mvin3(solver->work->g.data + i*DIM*DIM, temp_spad, DIM, remainder);
    //     gemmini_extended_mvin3(solver->work->x.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
    //     gemmini_extended_mvin3(solver->work->x_max.data + i*DIM*DIM, temp_spad + 5*DIM, DIM, remainder);
    //     gemmini_extended_mvin(solver->work->x_min.data + i*DIM*DIM, temp_spad + 3*DIM, DIM, remainder);
    //     gemmini_extended_preload(temp_spad, temp_spad+4*DIM, 4, 4, 4, 4); 
    //     gemmini_compute_preloaded(I_spad,   temp_spad + 1*DIM); 
    //     gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM,         temp_spad+ 4*DIM, DIM, 1);
    //     gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + DIM,   temp_spad+ 4*DIM+1, DIM, 1);
    //     gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + 2*DIM, temp_spad+ 4*DIM+2, DIM, 1);
    //     gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + 3*DIM, temp_spad+ 4*DIM+3, DIM, 1);
    //     // exit(0);
    // }
    // remainder = ((NHORIZON * NSTATES) / DIM) % DIM;
    //     gemmini_extended_mvin3(solver->work->g.data + i*DIM*DIM, temp_spad, DIM, remainder);
    //     gemmini_extended_mvin3(solver->work->x.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
    //     gemmini_extended_mvin3(solver->work->x_max.data + i*DIM*DIM, temp_spad + 5*DIM, DIM, remainder);
    //     gemmini_extended_mvin(solver->work->x_min.data + i*DIM*DIM, temp_spad + 3*DIM, DIM, remainder);
    //     gemmini_extended_preload(temp_spad, temp_spad+4*DIM, 4, 4, 4, 4); 
    //     gemmini_compute_preloaded(I_spad,   temp_spad + 1*DIM); 
    //     gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM,         temp_spad+ 4*DIM, DIM, 1);
    //     gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + DIM,   temp_spad+ 4*DIM+1, DIM, 1);
    //     // gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + 2*DIM, temp_spad+ 4*DIM+2, DIM, 1);
    //     // gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + 3*DIM, temp_spad+ 4*DIM+3, DIM, 1);

    // gemmini_fence();
    // printf("vnew:  (temp)");
    // for(int j = 0; j < (NHORIZON) * NSTATES; j++) {
    //     printf("%0.5f ", solver->work->vnew.data[j]);
    // }
    // printf("\n");
    // for(i = 0; i < (NHORIZON * NSTATES) / (DIM * DIM); i++) {
    //     gemmini_extended_mvin3(solver->work->vnew.data + i*DIM*DIM, temp_spad + 2*DIM, DIM, remainder);
    //     gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM,         temp_spad+ 2*DIM, DIM, 1);
    //     gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + DIM,   temp_spad+ 2*DIM+1, DIM, 1);
    //     gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + 2*DIM, temp_spad+ 2*DIM+2, DIM, 1);
    //     gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + 3*DIM, temp_spad+ 2*DIM+3, DIM, 1);
    // }
    // remainder = ((NHORIZON * NSTATES) / DIM) % DIM;
    // gemmini_extended_mvin3(solver->work->vnew.data + i*DIM*DIM, temp_spad + 2*DIM, DIM, remainder);
    // gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM,         temp_spad+ 2*DIM, DIM, 1);
    // gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + DIM,   temp_spad+ 2*DIM+1, DIM, 1);
    // gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + 2*DIM, temp_spad+ 2*DIM+2, DIM, 1);
    // gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM + 3*DIM, temp_spad+ 2*DIM+3, DIM, 1);

    // gemmini_extended2_config_st(DIM*sizeof(float), 0, 1.0, 1, 2, DIM, 1, 1, 2, 4, 0, 1);
    gemmini_extended_config_ex(OUTPUT_STATIONARY, RELU, 0, 1, false, false);
    gemmini_extended_config_st(DIM*sizeof(float), 0, 1.0);
    gemmini_extended3_config_ld(sizeof(float) * DIM, 1.0, false, 0);
    gemmini_extended3_config_ld(sizeof(float) * DIM, -1.0, false, 2);


    size_t i;
    uint8_t remainder;
    // TODO HANDLE IF

    const uint32_t temp_g_spad = temp_spad + 0;
    const uint32_t temp_x_spad = temp_spad + 1*DIM;
    const uint32_t temp_x_max_spad = temp_spad + 5*DIM;
    const uint32_t temp_x_min_spad = temp_spad + 3*DIM;

    const uint32_t temp_clip_spad = temp_spad + 2*DIM;

    remainder = DIM;
    for(i = 0; i < (NHORIZON * NSTATES) / (DIM * DIM); i++) {
        gemmini_extended_config_ex(OUTPUT_STATIONARY, RELU, 0, 1, false, false);
        gemmini_extended_mvin3(solver->work->g.data + i*DIM*DIM, temp_g_spad, DIM, remainder);
        gemmini_extended_mvin3(solver->work->x.data + i*DIM*DIM, temp_x_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->x_max.data + i*DIM*DIM, temp_x_max_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->x_min.data + i*DIM*DIM, temp_x_min_spad, DIM, remainder);

        gemmini_extended_preload(temp_g_spad, GARBAGE_ADDR, 4, 4, 4, 4); 
        gemmini_compute_preloaded(I_spad,   temp_x_spad);  // add inputs, negated
        gemmini_extended_preload(GARBAGE_ADDR, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_accumulated(I_spad,   temp_x_max_spad);  // add the maximum


        gemmini_extended_preload(temp_x_max_spad, GARBAGE_ADDR, 4, 4, 4, 4); 
        gemmini_compute_preloaded(nI_spad,   temp_x_min_spad);  // subtract the minimum 
        gemmini_extended_preload(GARBAGE_ADDR, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_accumulated(nI_spad,   temp_clip_spad);  // 

        gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
        gemmini_extended_preload(temp_x_min_spad, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_preloaded(I_spad,   temp_clip_spad);  // add inputs, negated

        gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM, temp_clip_spad, DIM, remainder);

        // exit(0);
    }
    remainder = ((NHORIZON * NSTATES) / DIM) % DIM;
        gemmini_extended_config_ex(OUTPUT_STATIONARY, RELU, 0, 1, false, false);
        gemmini_extended_mvin3(solver->work->g.data + i*DIM*DIM, temp_g_spad, DIM, remainder);
        gemmini_extended_mvin3(solver->work->x.data + i*DIM*DIM, temp_x_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->x_max.data + i*DIM*DIM, temp_x_max_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->x_min.data + i*DIM*DIM, temp_x_min_spad, DIM, remainder);

        gemmini_extended_preload(temp_g_spad, GARBAGE_ADDR, 4, 4, 4, 4); 
        gemmini_compute_preloaded(I_spad,   temp_x_spad);  // add inputs, negated
        gemmini_extended_preload(GARBAGE_ADDR, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_accumulated(I_spad,   temp_x_max_spad);  // add the maximum


        gemmini_extended_preload(temp_x_max_spad, GARBAGE_ADDR, 4, 4, 4, 4); 
        gemmini_compute_preloaded(nI_spad,   temp_x_min_spad);  // subtract the minimum 
        gemmini_extended_preload(GARBAGE_ADDR, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_accumulated(nI_spad,   temp_clip_spad);  // 

        gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
        gemmini_extended_preload(temp_x_min_spad, temp_clip_spad, 4, 4, 4, 4); 
        gemmini_compute_preloaded(I_spad,   temp_clip_spad);  // add inputs, negated

        gemmini_extended_mvout(solver->work->vnew.data + i*DIM*DIM, temp_clip_spad, DIM, remainder);
    gemmini_fence();
    // printf("vnew:  ");
    // for(int j = 0; j < (NHORIZON) * NSTATES; j++) {
    //     printf("%0.5f ", solver->work->vnew.data[j]);
    // }
    // printf("\n");

    // matadd(solver->work->x.data, solver->work->g.data, solver->work->vnew.data, NHORIZON, NSTATES);
    // if (solver->settings->en_state_bound) {
    //     cwisemax(solver->work->x_min.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
    //     cwisemin(solver->work->x_max.data, solver->work->s1.data, solver->work->vnew.data, NHORIZON, NSTATES);
    // }
    // printf("vnew:  ");
    // for(int j = 0; j < (NHORIZON) * NSTATES; j++) {
    //     printf("%0.5f ", solver->work->vnew.data[j]);
    // }
    // printf("\n");
    // exit(0);
    
    // matadd(solver->work->x.data, solver->work->g.data, solver->work->vnew.data, NHORIZON, NSTATES);
    // if (solver->settings->en_state_bound) {
    //     cwisemax(solver->work->x_min.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
    //     cwisemin(solver->work->x_max.data, solver->work->s1.data, solver->work->vnew.data, NHORIZON, NSTATES);
    // }
    #else
    matadd(solver->work->x.data, solver->work->g.data, solver->work->vnew.data, NHORIZON, NSTATES);
    if (solver->settings->en_state_bound) {
        cwisemax(solver->work->x_min.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
        cwisemin(solver->work->x_max.data, solver->work->s1.data, solver->work->vnew.data, NHORIZON, NSTATES);
    }
    #endif
    TRACE_CHECKSUM(update_slack_2, solver->work->vnew);
}


inline void primal_residual_state(TinySolver *solver) {
    #ifdef USE_GEMMINI
    gemmini_extended_config_ex(OUTPUT_STATIONARY, RELU, 0, 1, false, false);
    gemmini_extended_config_st(DIM*sizeof(float), 0, 1.0);
    gemmini_extended3_config_ld(sizeof(float) * DIM, 1.0, false, 0);

    // gemmini_extended_preload(GARBAGE_ADDR, nrI_spad, 4, 4, 4, 4);
    // gemmini_compute_preloaded(rI_spad,   I_spad);
    // gemmini_extended_mvout(nrI, nrI_spad, 4, 4);
    // gemmini_fence();
    size_t i;
    uint8_t remainder;



    remainder = DIM;
    // temp_spad + 2*DIM: pos_relu
    // temp_spad + 3*DIM: neg_relu
    // temp_spad + 4*DIM: abs_relu 
    for(i = 0; i < (NHORIZON * NSTATES) / (DIM * DIM); i++) {
        gemmini_extended_mvin(solver->work->x.data + i*DIM*DIM, temp_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->vnew.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
        gemmini_extended_preload(temp_spad, temp_spad+2*DIM, 4, 4, 4, 4); // preload x
        gemmini_compute_preloaded(nI_spad,   temp_spad + 1*DIM); // vnew
        gemmini_extended_preload(temp_spad+1*DIM, temp_spad+3*DIM, 4, 4, 4, 4); // preload vnew
        gemmini_compute_preloaded(nI_spad,   temp_spad); // x
        gemmini_extended_preload(temp_spad+2*DIM, temp_spad+4*DIM, 4, 4, 4, 4); // preload vnew
        gemmini_compute_preloaded(I_spad,   temp_spad+3*DIM); // x
        gemmini_extended_mvout(solver->work->s2.data + i*DIM*DIM, temp_spad+ 2*DIM, DIM, remainder);
    }
    remainder = ((NHORIZON * NSTATES) / DIM) % DIM;
    gemmini_extended_mvin(solver->work->x.data + i*DIM*DIM, temp_spad , DIM, remainder);
    gemmini_extended_mvin(solver->work->vnew.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
    gemmini_extended_preload(temp_spad, temp_spad+2*DIM, 4, 4, 4, 4); // preload x
    gemmini_compute_preloaded(nI_spad,   temp_spad + 1*DIM); // vnew
    gemmini_extended_preload(temp_spad+1*DIM, temp_spad+3*DIM, 4, 4, 4, 4); // preload vnew
    gemmini_compute_preloaded(nI_spad,   temp_spad); // x
    gemmini_extended_preload(temp_spad+2*DIM, temp_spad+4*DIM, 4, 4, 4, 4); // preload vnew
    gemmini_compute_preloaded(I_spad,   temp_spad+3*DIM); // x
    gemmini_extended_mvout(solver->work->s2.data + i*DIM*DIM, temp_spad+ 2*DIM, DIM, remainder);
    gemmini_fence();



    solver->work->primal_residual_state = maxcoeff(solver->work->s2.data, NHORIZON, NSTATES);
    #else
    matsub(solver->work->x.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
    cwiseabs(solver->work->s1.data, solver->work->s2.data, NHORIZON, NSTATES);
    solver->work->primal_residual_state = maxcoeff(solver->work->s2.data, NHORIZON, NSTATES);
    TRACE_CHECKSUM(primal_residual_state, solver->work->s2);
    #endif
}

inline void dual_residual_state(TinySolver *solver) {
    #ifdef USE_GEMMINI
    // gemmini_extended_preload(GARBAGE_ADDR, nrI_spad, 4, 4, 4, 4);
    // gemmini_compute_preloaded(rI_spad,   I_spad);
    // gemmini_extended_mvout(nrI, nrI_spad, 4, 4);
    // gemmini_fence();
    size_t i;
    uint8_t remainder;

    gemmini_extended_config_ex(OUTPUT_STATIONARY, RELU, 0, 1, false, false);
    gemmini_extended_config_st(DIM*sizeof(float), 0, 1.0);
    gemmini_extended3_config_ld(sizeof(float) * DIM, solver->cache->rho, false, 0);


    remainder = DIM;
    // temp_spad + 2*DIM: pos_relu
    // temp_spad + 3*DIM: neg_relu
    // temp_spad + 4*DIM: abs_relu 
    for(i = 0; i < (NHORIZON * NSTATES) / (DIM * DIM); i++) {
        gemmini_extended_mvin(solver->work->v.data + i*DIM*DIM, temp_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->vnew.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
        gemmini_extended_preload(temp_spad, temp_spad+2*DIM, 4, 4, 4, 4); // preload v
        gemmini_compute_preloaded(nI_spad,   temp_spad + 1*DIM); // vnew
        gemmini_extended_preload(temp_spad+1*DIM, temp_spad+3*DIM, 4, 4, 4, 4); // preload vnew
        gemmini_compute_preloaded(nI_spad,   temp_spad); // v
        gemmini_extended_preload(temp_spad+2*DIM, temp_spad+4*DIM, 4, 4, 4, 4); // preload vnew
        gemmini_compute_preloaded(I_spad,   temp_spad+3*DIM); // v
        gemmini_extended_mvout(solver->work->s2.data + i*DIM*DIM, temp_spad+ 2*DIM, DIM, remainder);
    }
    remainder = ((NHORIZON * NSTATES) / DIM) % DIM;
    gemmini_extended_mvin(solver->work->v.data + i*DIM*DIM, temp_spad , DIM, remainder);
    gemmini_extended_mvin(solver->work->vnew.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
    gemmini_extended_preload(temp_spad, temp_spad+2*DIM, 4, 4, 4, 4); // preload x
    gemmini_compute_preloaded(nI_spad,   temp_spad + 1*DIM); // vnew
    gemmini_extended_preload(temp_spad+1*DIM, temp_spad+3*DIM, 4, 4, 4, 4); // preload vnew
    gemmini_compute_preloaded(nI_spad,   temp_spad); // v
    gemmini_extended_preload(temp_spad+2*DIM, temp_spad+4*DIM, 4, 4, 4, 4); // preload vnew
    gemmini_compute_preloaded(I_spad,   temp_spad+3*DIM); // v
    gemmini_extended_mvout(solver->work->s2.data + i*DIM*DIM, temp_spad+ 2*DIM, DIM, remainder);

    gemmini_fence();
    solver->work->dual_residual_state = maxcoeff(solver->work->s2.data, NHORIZON, NSTATES);
    #else
    matsub(solver->work->v.data, solver->work->vnew.data, solver->work->s1.data, NHORIZON, NSTATES);
    cwiseabs(solver->work->s1.data, solver->work->s2.data, NHORIZON, NSTATES);
    solver->work->dual_residual_state = maxcoeff(solver->work->s2.data, NHORIZON, NSTATES) * solver->cache->rho;
    TRACE_CHECKSUM(dual_residual_state, solver->work->s2);
    #endif
}

inline void primal_residual_input(TinySolver *solver) {
    #ifdef USE_GEMMINI
    gemmini_extended_config_ex(OUTPUT_STATIONARY, RELU, 0, 1, false, false);
    gemmini_extended_config_st(DIM*sizeof(float), 0, 1.0);
    gemmini_extended3_config_ld(sizeof(float) * DIM, 1.0, false, 0);

    size_t i;
    uint8_t remainder;



    remainder = DIM;
    for(i = 0; i < ((NHORIZON-1) * NINPUTS) / (DIM * DIM); i++) {
        gemmini_extended_mvin(solver->work->u.data + i*DIM*DIM, temp_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->znew.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
        gemmini_extended_preload(temp_spad, temp_spad+2*DIM, 4, 4, 4, 4); // preload x
        gemmini_compute_preloaded(nI_spad,   temp_spad + 1*DIM); // vnew
        gemmini_extended_preload(temp_spad+1*DIM, temp_spad+3*DIM, 4, 4, 4, 4); // preload vnew
        gemmini_compute_preloaded(nI_spad,   temp_spad); // x
        gemmini_extended_preload(temp_spad+2*DIM, temp_spad+4*DIM, 4, 4, 4, 4); // preload vnew
        gemmini_compute_preloaded(I_spad,   temp_spad+3*DIM); // x
        gemmini_extended_mvout(solver->work->m2.data + i*DIM*DIM, temp_spad+ 2*DIM, DIM, remainder);
    }
    remainder = (((NHORIZON-1) * NINPUTS) / DIM) % DIM;
    gemmini_extended_mvin(solver->work->u.data + i*DIM*DIM, temp_spad , DIM, remainder);
    gemmini_extended_mvin(solver->work->znew.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
    gemmini_extended_preload(temp_spad, temp_spad+2*DIM, 4, 4, 4, 4); // preload x
    gemmini_compute_preloaded(nI_spad,   temp_spad + 1*DIM); // vnew
    gemmini_extended_preload(temp_spad+1*DIM, temp_spad+3*DIM, 4, 4, 4, 4); // preload vnew
    gemmini_compute_preloaded(nI_spad,   temp_spad); // x
    gemmini_extended_preload(temp_spad+2*DIM, temp_spad+4*DIM, 4, 4, 4, 4); // preload vnew
    gemmini_compute_preloaded(I_spad,   temp_spad+3*DIM); // x
    gemmini_extended_mvout(solver->work->m2.data + i*DIM*DIM, temp_spad+ 2*DIM, DIM, remainder);
    gemmini_fence();
    solver->work->primal_residual_input = maxcoeff(solver->work->m2.data, NHORIZON - 1, NINPUTS);
    #else

    matsub(solver->work->u.data, solver->work->znew.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    cwiseabs(solver->work->m1.data, solver->work->m2.data, NHORIZON - 1, NINPUTS);
    solver->work->primal_residual_input = maxcoeff(solver->work->m2.data, NHORIZON - 1, NINPUTS);
    TRACE_CHECKSUM(primal_residual_input, solver->work->m2);
    #endif
}

inline void dual_residual_input(TinySolver *solver) {
    #ifdef USE_GEMMINI
    gemmini_extended_config_ex(OUTPUT_STATIONARY, RELU, 0, 1, false, false);
    gemmini_extended_config_st(DIM*sizeof(float), 0, 1.0);
    gemmini_extended3_config_ld(sizeof(float) * DIM, 1.0, false, 0);

    size_t i;
    uint8_t remainder;



    remainder = DIM;
    for(i = 0; i < ((NHORIZON-1) * NINPUTS) / (DIM * DIM); i++) {
        gemmini_extended_mvin(solver->work->z.data + i*DIM*DIM, temp_spad, DIM, remainder);
        gemmini_extended_mvin(solver->work->znew.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
        gemmini_extended_preload(temp_spad, temp_spad+2*DIM, 4, 4, 4, 4); // preload x
        gemmini_compute_preloaded(nI_spad,   temp_spad + 1*DIM); // vnew
        gemmini_extended_preload(temp_spad+1*DIM, temp_spad+3*DIM, 4, 4, 4, 4); // preload vnew
        gemmini_compute_preloaded(nI_spad,   temp_spad); // x
        gemmini_extended_preload(temp_spad+2*DIM, temp_spad+4*DIM, 4, 4, 4, 4); // preload vnew
        gemmini_compute_preloaded(I_spad,   temp_spad+3*DIM); // x
        gemmini_extended_mvout(solver->work->m2.data + i*DIM*DIM, temp_spad+ 2*DIM, DIM, remainder);
    }
    remainder = (((NHORIZON-1) * NINPUTS) / DIM) % DIM;
    gemmini_extended_mvin(solver->work->z.data + i*DIM*DIM, temp_spad , DIM, remainder);
    gemmini_extended_mvin(solver->work->znew.data + i*DIM*DIM, temp_spad + 1*DIM, DIM, remainder);
    gemmini_extended_preload(temp_spad, temp_spad+2*DIM, 4, 4, 4, 4); // preload x
    gemmini_compute_preloaded(nI_spad,   temp_spad + 1*DIM); // vnew
    gemmini_extended_preload(temp_spad+1*DIM, temp_spad+3*DIM, 4, 4, 4, 4); // preload vnew
    gemmini_compute_preloaded(nI_spad,   temp_spad); // x
    gemmini_extended_preload(temp_spad+2*DIM, temp_spad+4*DIM, 4, 4, 4, 4); // preload vnew
    gemmini_compute_preloaded(I_spad,   temp_spad+3*DIM); // x
    gemmini_extended_mvout(solver->work->m2.data + i*DIM*DIM, temp_spad+ 2*DIM, DIM, remainder);
    gemmini_fence();
    solver->work->primal_residual_input = maxcoeff(solver->work->m2.data, NHORIZON - 1, NINPUTS);
    #else

    matsub(solver->work->z.data, solver->work->znew.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    cwiseabs(solver->work->m1.data, solver->work->m2.data, NHORIZON - 1, NINPUTS);
    solver->work->dual_residual_input = maxcoeff(solver->work->m2.data, NHORIZON - 1, NINPUTS) * solver->cache->rho;
    TRACE_CHECKSUM(dual_residual_input, solver->work->m2);
    #endif
}

inline void update_linear_cost_1(TinySolver *solver) {
    #ifdef USE_GEMMINI
    gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 1, false, false);
    gemmini_config_st(4);
    gemmini_extended3_config_ld(4, 1.0, false, 1);
    for(size_t i = 0; i < NHORIZON-1; i++) {
        gemmini_extended_mvin2(solver->work->znew.col(i), znew_spad + i*NINPUTS, 1, DIM);
        gemmini_extended_mvin2(solver->work->y.col(i), y_spad + i*NINPUTS, 1, DIM);

        gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
        gemmini_compute_preloaded(nrI_spad,   znew_spad + i*NINPUTS);
        gemmini_extended_preload(GARBAGE_ADDR, r_spad + i*NINPUTS, 1, 4, 1, 4);
        gemmini_compute_accumulated(rI_spad,   y_spad + i*NINPUTS);
        gemmini_extended_mvout(solver->work->r.data + i*NINPUTS, r_spad + i*NINPUTS, 1, 4);
    }
    gemmini_fence();
    // printf("r: ");
    // for(int j = 0; j < NINPUTS; j++) {
    //     printf("%0.5f ", solver->work->r.data[j]);
    // }
    // printf("\n");
    // matsub(solver->work->znew.data, solver->work->y.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    // matmulf(solver->work->m1.data, solver->work->r.data, -solver->cache->rho, NHORIZON - 1, NINPUTS);
    // printf("r: ");
    // for(int j = 0; j < NINPUTS; j++) {
    //     printf("%0.5f ", solver->work->r.data[j]);
    // }
    // printf("\n");
    // exit(0);
    #else
    matsub(solver->work->znew.data, solver->work->y.data, solver->work->m1.data, NHORIZON - 1, NINPUTS);
    matmulf(solver->work->m1.data, solver->work->r.data, -solver->cache->rho, NHORIZON - 1, NINPUTS);
    TRACE_CHECKSUM(update_linear_cost_1, solver->work->r);
    #endif
}

inline void update_linear_cost_2(TinySolver *solver, int i) {
    cwisemul(solver->work->Xref.col(i), solver->work->Q.data, solver->work->x1.data, 1, NSTATES);
    matneg(solver->work->x1.data, solver->work->q.col(i), 1, NSTATES);
    TRACE_CHECKSUM(update_linear_cost_2, solver->work->q);
}

inline void update_linear_cost_3(TinySolver *solver) {
    #ifdef USE_GEMMINI
        for(size_t i = 0; i < NHORIZON-1; i++) {
            gemmini_extended_mvin2(solver->work->vnew.col(i), vnew_spad + i*NSTATES, 1, DIM);
            gemmini_extended_mvin2(solver->work->vnew.col(i) + DIM, vnew_spad + i*NSTATES + DIM, 1, DIM);
            gemmini_extended_mvin2(solver->work->vnew.col(i) + 2*DIM, vnew_spad + i*NSTATES + 2*DIM, 1, DIM);

            gemmini_extended_mvin2(solver->work->g.col(i), g_spad + i*NSTATES, 1, DIM);
            gemmini_extended_mvin2(solver->work->g.col(i) + DIM, g_spad + i*NSTATES + DIM, 1, DIM);
            gemmini_extended_mvin2(solver->work->g.col(i) + 2*DIM, g_spad + i*NSTATES + 2*DIM, 1, DIM);

            gemmini_extended_mvin2(solver->work->q.col(i), q_spad + i*NSTATES, 1, DIM);
            gemmini_extended_mvin2(solver->work->q.col(i) + DIM, q_spad + i*NSTATES + DIM, 1, DIM);
            gemmini_extended_mvin2(solver->work->q.col(i) + 2*DIM, q_spad + i*NSTATES + 2*DIM, 1, DIM);

            for(size_t j = 0; j < 3; j++) {
                gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
                gemmini_compute_preloaded(nrI_spad,   vnew_spad + i*NSTATES + j*DIM);
                gemmini_extended_preload(GARBAGE_ADDR, GARBAGE_ADDR, 1, 4, 1, 4);
                gemmini_compute_accumulated(nI_spad,   g_spad + i*NSTATES + j*DIM);
                gemmini_extended_preload(GARBAGE_ADDR, q_spad + i*NSTATES + j*DIM, 1, 4, 1, 4);
                gemmini_compute_accumulated(I_spad,   q_spad + i*NSTATES + j*DIM);
                gemmini_extended_mvout(solver->work->q.data + i*NSTATES + j*DIM, q_spad + i*NSTATES + j*DIM, 1, 4);
            }

        }   
        gemmini_fence();
    // printf("q: ");
    // for(int j = 0; j < NSTATES; j++) {
    //     printf("%0.5f ", solver->work->q.data[j]);
    // }
    // printf("\n");
    //     matsub(solver->work->vnew.data, solver->work->g.data, solver->work->s1.data, NHORIZON, NSTATES);
    //     matmulf(solver->work->s1.data, solver->work->s2.data, solver->cache->rho, NHORIZON, NSTATES);
    //     matsub(solver->work->q.data, solver->work->s2.data, solver->work->s1.data, NHORIZON, NSTATES);
    //     solver->work->q.set(solver->work->s1.data);
    // printf("q: ");
    // for(int j = 0; j < NSTATES; j++) {
    //     printf("%0.5f ", solver->work->q.data[j]);
    // }
    // printf("\n");
    // exit(0);
    #else
        matsub(solver->work->vnew.data, solver->work->g.data, solver->work->s1.data, NHORIZON, NSTATES);
        matmulf(solver->work->s1.data, solver->work->s2.data, solver->cache->rho, NHORIZON, NSTATES);
        matsub(solver->work->q.data, solver->work->s2.data, solver->work->s1.data, NHORIZON, NSTATES);
        solver->work->q.set(solver->work->s1.data);
        TRACE_CHECKSUM(update_linear_cost_3, solver->work->s1);
    #endif
}

inline void update_linear_cost_4(TinySolver *solver) {
    matsub(solver->work->vnew.col(NHORIZON - 1), solver->work->g.col(NHORIZON - 1), solver->work->x1.data, 1, NSTATES);
    matmulf(solver->work->x1.data, solver->work->x2.data, solver->cache->rho, 1, NSTATES);
#ifdef USE_MATVEC
    matvec(solver->cache->PinfT.data, solver->work->Xref.col(NHORIZON - 1), solver->work->x1.data, NSTATES, NSTATES);
#else
    matmul(solver->work->Xref.col(NHORIZON - 1), solver->cache->PinfT.data, solver->work->x1.data, 1, NSTATES, NSTATES);
#endif
    matadd(solver->work->x1.data, solver->work->x2.data, solver->work->x3.data, 1, NSTATES);
    matneg(solver->work->x3.data, solver->work->p.col(NHORIZON - 1), 1, NSTATES);
    TRACE_CHECKSUM(update_linear_cost_4, solver->work->p);
}

/**
 * Update linear terms from Riccati backward pass
 */
inline void backward_pass(TinySolver *solver)
{
    // printf("Starting backward pass!\n");
    #ifdef USE_GEMMINI
    gemmini_extended3_config_ld(4, 1.0, false, 1);
    gemmini_extended_mvin2(solver->work->p.col(NHORIZON-1), p_spad + (NHORIZON-1)*NSTATES, 1, DIM);
    gemmini_extended_mvin2(solver->work->p.col(NHORIZON-1) + DIM, p_spad + (NHORIZON-1)*NSTATES + DIM, 1, DIM);
    gemmini_extended_mvin2(solver->work->p.col(NHORIZON-1) + 2*DIM, p_spad + (NHORIZON-1)*NSTATES + 2*DIM, 1, DIM);
    #endif
    for (int i = NHORIZON - 2; i >= 0; i--) {
        backward_pass_1(solver, i);
        backward_pass_2(solver, i);
    }
    #ifdef USE_GEMMINI
    gemmini_fence();
    #endif
}

/**
 * Use LQR feedback policy to roll out trajectory
 */
inline void forward_pass(TinySolver *solver)
{
    // printf("Starting forward pass!\n");
    #ifdef USE_GEMMINI
    gemmini_extended3_config_ld(4, 1.0, false, 1);
    gemmini_extended_mvin2(solver->work->x.col(0), x_spad , 1, DIM);
    gemmini_extended_mvin2(solver->work->x.col(0) + DIM, x_spad + DIM, 1, DIM);
    gemmini_extended_mvin2(solver->work->x.col(0) + 2*DIM, x_spad + 2*DIM, 1, DIM);
    #endif
    for (int i = 0; i < NHORIZON - 1; i++) {
        forward_pass_1(solver, i);
        forward_pass_2(solver, i);
    }
    #ifdef USE_GEMMINI
    gemmini_fence();
    #endif
}

/**
 * Do backward Riccati pass then forward roll out
 */
inline void update_primal(TinySolver *solver)
{
    CYCLE_CNT_WRAPPER(backward_pass, solver, "update_primal_backward_pass");
    CYCLE_CNT_WRAPPER(forward_pass, solver, "update_primal_forward_pass");
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint projection function
 */
inline void update_slack(TinySolver *solver)
{
    update_slack_1(solver);
    update_slack_2(solver);
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
 */
inline void update_dual(TinySolver *solver)
{
    update_dual_1(solver);
}


/**
 * Update linear control cost terms in the Riccati feedback using
 * the changing slack and dual variables from ADMM
 */
inline void update_linear_cost(TinySolver *solver)
{
    update_linear_cost_1(solver);
    for (int i = 0; i < NHORIZON; i++) {
        update_linear_cost_2(solver, i);
    }
    update_linear_cost_3(solver);
    update_linear_cost_4(solver);
}

void tiny_init(TinySolver *solver);


// inline void tiny_init(TinySolver *solver) {

// }

};
#endif //TINYMPC_ADMM_RVV_HPP