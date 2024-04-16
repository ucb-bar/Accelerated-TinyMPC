#pragma once
#ifndef RVV_MATLIB_COMMON_H
#define RVV_MATLIB_COMMON_H

#define MSTATUS_VS          0x00000600
#define MSTATUS_FS          0x00006000
#define MSTATUS_XS          0x00018000

extern "C" {

#include <stdint.h>

void gen_rand_1d(float *a, int n);
void gen_string(char *s, int n);
void gen_rand_2d(float **ar, int n, int m);
void print_string(const char *a, const char *name);
void print_array_1d(float *a, int n, const char *type, const char *name);
void print_array_2d(float **a, int n, int m, const char *type, const char *name);
bool float_eq(float golden, float actual, float relErr);
bool compare_1d(float *golden, float *actual, int n);
bool compare_string(const char *golden, const char *actual, int n);
bool compare_2d(float **golden, float **actual, int n, int m);
float **alloc_array_2d(int n, int m);
float **alloc_array_2d_col(int n, int m);
float *alloc_array_1d(int n);
void free_array_2d(float **ar);
void free_array_1d(float *ar);
void init_array_zero_1d(float *ar, int n);
void init_array_one_1d(float *ar, int n);
void init_array_one_2d(float **ar, int n, int m);

void printx(float **a, int n, int m, const char *name);

#ifdef USE_PK
static inline void enable_vector_operations() {
    printf("Using PK\n");
}
#else
static inline void enable_vector_operations() {
    unsigned long mstatus;

    // Read current mstatus
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));

    // Set VS field to Dirty (11)
    mstatus |= MSTATUS_VS | MSTATUS_FS | MSTATUS_XS;

    // Write back updated mstatus
    asm volatile("csrw mstatus, %0" :: "r"(mstatus));
}
#endif // USE_PK

static uint64_t read_cycles() {
    uint64_t cycles;
    // asm volatile ("rdcycle %0" : "=r" (cycles));
    asm volatile ("csrr %0, cycle" : "=r" (cycles));
    return cycles;
}

};
#endif
