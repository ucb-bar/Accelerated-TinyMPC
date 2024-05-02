#pragma once
#ifndef TINYMPC_MATLIB_RVA_H
#define TINYMPC_MATLIB_RVA_H

#include "matlib_rvv.h"

extern "C"
{

// RVV-specific function declarations in array format
float maxcoeff_rva(float **a, int n, int m);
float mincoeff_rva(float **a, int n, int m);
float matnorm_rva(float **a, int n, int m);
void matneg_rva(float **a, float **b, int n, int m);
void cwiseabs_rva(float **a, float **b, int n, int m);
void cwisemin_rva(float **a, float **b, float **c, int n, int m);
void cwisemax_rva(float **a, float **b, float **c, int n, int m);
void cwisemul_rva(float **a, float **b, float **c, int n, int m);
void matmul_rva(float **a, float **b, float **c, int n, int m, int o);
void matvec_rva(float **a, float **b, float **c, int n, int m);
void matvec_transpose_rva(float **a, float **b, float **c, int n, int m);
void matmulf_rva(float **a, float **b, float f, int n, int m);
void matsub_rva(float **a, float **b, float **c, int n, int m);
void matadd_rva(float **a, float **b, float **c, int n, int m);
void transpose_rva(float **a, float **b, int n, int m);
void matcopy_rva(float **a, float **b, int n, int m);
void matset_rva(float **a, float f, int n, int m);
void matsetv_rva(float **a, float *f, int n, int m);

#ifdef USE_RVA
#define maxcoeff maxcoeff_rva
#define mincoeff mincoeff_rva
#define matnorm matnorm_rva
#define matneg matneg_rva
#define cwiseabs cwiseabs_rva
#define cwisemin cwisemin_rva
#define cwisemax cwisemax_rva
#define cwisemul cwisemul_rva
#define matmul matmul_rva
#define matvec matvec_rva
#define matvec_transpose matvec_transpose_rva
#define matmulf matmulf_rva
#define matsub matsub_rva
#define matadd matadd_rva
#define transpose transpose_rva
#define matcopy matcopy_rva
#define matset matset_rva
#define matsetv matsetv_rva
#endif

// matrix maximum coefficient
inline float maxcoeff_rva(float **a, int n, int m) {
    return maxcoeff_rvv(a[0], n, m);
}

// matrix min coefficient
inline float mincoeff_rva(float **a, int n, int m) {
    return mincoeff_rvv(a[0], n, m);
}

// matrix unary negative
inline void matneg_rva(float **a, float **b, int n, int m) {
    matneg_rvv(a[0], b[0], n, m);
}

// matrix coefficient-wise abs
inline void cwiseabs_rva(float **a, float **b, int n, int m) {
    cwiseabs_rvv(a[0], b[0], n, m);
}

// matrix coefficient-wise min
inline void cwisemin_rva(float **a, float **b, float **c, int n, int m) {
    cwisemin_rvv(a[0], b[0], c[0], n, m);
}

// matrix coefficient-wise multiplication
inline void cwisemul_rva(float **a, float **b, float **c, int n, int m) {
    cwisemul_rvv(a[0], b[0], c[0], n, m);
}

// matrix coefficient-wise max
inline void cwisemax_rva(float **a, float **b, float **c, int n, int m) {
    cwisemax_rvv(a[0], b[0], c[0], n, m);
}

// matrix multiplication, note B is not [o][m]
// A[n][o], B[m][o] --> C[n][m];
inline void matmul_rva(float **a, float **b, float **c, int n, int m, int o) {
    matmul_rvv(a[0], b[0], c[0], n, m, o);
}

/*  a is row major
 *        j
 *    1 2 3 4     9
 *  i 5 6 7 8  *  6
 *    9 8 7 6     5 j
 *                4
 */
inline void matvec_rva(float **a, float **b, float **c, int n, int m) {
    matvec_rvv(a[0], b[0], c[0], n, m);
}

/*  a is col major
 *      j         i
 *  9 6 5 4  *  1 5 9
 *              2 6 8
 *              3 7 7 j
 *              4 8 6
 */
inline void matvec_transpose_rva(float **a, float **b, float **c, int n, int m) {
    matvec_transpose_rvv(a[0], b[0], c[0], n, m);
}

// matrix scalar multiplication
inline void matmulf_rva(float **a, float **b, float f, int n, int m) {
    matmulf_rvv(a[0], b[0], f, n, m);
}

// matrix subtraction
inline void matsub_rva(float **a, float **b, float **c, int n, int m) {
    matsub_rvv(a[0], b[0], c[0], n, m);
}

// matrix addition
inline void matadd_rva(float **a, float **b, float **c, int n, int m) {
    matsub_rvv(a[0], b[0], c[0], n, m);
}

// matrix transpose
inline void transpose_rva(float **a, float **b, int n, int m) {
    transpose_rvv(a[0], b[0], n, m);
};

// matrix copy
inline void matcopy_rva(float **a, float **b, int n, int m) {
    matcopy_rvv((const float *)(a[0]), b[0], n, m);
}

inline void matset_rva(float **a, float f, int n, int m) {
    matset_rvv(a[0], f, n, m);
}

inline void matsetv_rva(float **a, float *f, int n, int m) {
    matsetv_rvv(a[0], f, n, m);
}

// matrix l2 norm
inline float matnorm_rva(float **a, int n, int m) {
    return matnorm_rvv(a[0], n, m);
}

}
#endif //TINYMPC_MATLIB_RVA_H
