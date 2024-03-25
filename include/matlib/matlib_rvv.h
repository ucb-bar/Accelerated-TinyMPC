#pragma once
#ifndef TINYMPC_RVV_MATLIB_H
#define TINYMPC_RVV_MATLIB_H

extern "C"
{

float maxcoeff_golden(float **a, int n, int m);
float maxcoeff(float **a, int n, int m);
float mincoeff_golden(float **a, int n, int m);
float mincoeff(float **a, int n, int m);
float matnorm_golden(float **a, int n, int m);
float matnorm(float **a, int n, int m);
void matneg_golden(float **a, float **b, int n, int m);
void matneg(float **a, float **b, int n, int m);
void cwiseabs_golden(float **a, float **b, int n, int m);
void cwiseabs(float **a, float **b, int n, int m);
void cwisemin_golden(float **a, float **b, float **c, int n, int m);
void cwisemin(float **a, float **b, float **c, int n, int m);
void cwisemax_golden(float **a, float **b, float **c, int n, int m);
void cwisemax(float **a, float **b, float **c, int n, int m);
void cwisemul_golden(float **a, float **b, float **c, int n, int m);
void cwisemul(float **a, float **b, float **c, int n, int m);
void matmul_golden(float **a, float **b, float **c, int n, int m, int o);
void matmul(float **a, float **b, float **c, int n, int m, int o);
void matvec_golden(float **a, float **b, float **c, int n, int m);
void matvec(float **a, float **b, float **c, int n, int m);
void matvec_transpose_golden(float **a, float **b, float **c, int n, int m);
void matvec_transpose(float **a, float **b, float **c, int n, int m);
void matmulf_golden(float **a, float **b, float f, int n, int m);
void matmulf(float **a, float **b, float f, int n, int m);
void matsub_golden(float **a, float **b, float **c, int n, int m);
void matsub(float **a, float **b, float **c, int n, int m);
void matadd_golden(float **a, float **b, float **c, int n, int m);
void matadd(float **a, float **b, float **c, int n, int m);
void transpose_golden(float **a, float **b, int n, int m);
void transpose(float **a, float **b, int n, int m);
void matcopy_golden(float **a, float **b, int n, int m);
void matcopy(float **a, float **b, int n, int m);
void matset_golden(float **a, float f, int n, int m);
void matset(float **a, float f, int n, int m);
void matsetv_golden(float **a, float *f, int n, int m);
void matsetv(float **a, float *f, int n, int m);

}
#endif //TINYMPC_RVV_MATLIB_H
