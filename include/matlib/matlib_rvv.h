#pragma once
#ifndef TINYMPC_RVV_MATLIB_H
#define TINYMPC_RVV_MATLIB_H

#define USE_LMUL 1

#if USE_LMUL == 1
#define __riscv_vfabs_v_f32 __riscv_vfabs_v_f32m1
#define __riscv_vfadd_vv_f32 __riscv_vfadd_vv_f32m1
#define __riscv_vfmacc_vv_f32 __riscv_vfmacc_vv_f32m1
#define __riscv_vfmax_vv_f32 __riscv_vfmax_vv_f32m1
#define __riscv_vfmin_vv_f32 __riscv_vfmin_vv_f32m1
#define __riscv_vfmul_vf_f32 __riscv_vfmul_vf_f32m1
#define __riscv_vfmul_vv_f32 __riscv_vfmul_vv_f32m1
#define __riscv_vfmv_f_s_f32_f32 __riscv_vfmv_f_s_f32m1_f32
#define __riscv_vfmv_s_f_f32 __riscv_vfmv_s_f_f32m1
#define __riscv_vfmv_v_f_f32 __riscv_vfmv_v_f_f32m1
#define __riscv_vfneg_v_f32 __riscv_vfneg_v_f32m1
#define __riscv_vfredmax_vs_f32_f32 __riscv_vfredmax_vs_f32m1_f32m1
#define __riscv_vfredmin_vs_f32_f32 __riscv_vfredmin_vs_f32m1_f32m1
#define __riscv_vfredusum_vs_f32_f32 __riscv_vfredusum_vs_f32m1_f32m1
#define __riscv_vfredosum_vs_f32_f32 __riscv_vfredosum_vs_f32m1_f32m1
#define __riscv_vfsub_vv_f32 __riscv_vfsub_vv_f32m1
#define __riscv_vle32_v_f32 __riscv_vle32_v_f32m1
#define __riscv_vlse32_v_f32 __riscv_vlse32_v_f32m1
#define __riscv_vse32_v_f32 __riscv_vse32_v_f32m1
#define __riscv_vsetvl_e32 __riscv_vsetvl_e32m1
#define __riscv_vsetvlmax_e32 __riscv_vsetvlmax_e32m1
#define vfloat32_t vfloat32m1_t
#elif USE_LMUL == 2
#define __riscv_vfabs_v_f32 __riscv_vfabs_v_f32m2
#define __riscv_vfadd_vv_f32 __riscv_vfadd_vv_f32m2
#define __riscv_vfmacc_vv_f32 __riscv_vfmacc_vv_f32m2
#define __riscv_vfmax_vv_f32 __riscv_vfmax_vv_f32m2
#define __riscv_vfmin_vv_f32 __riscv_vfmin_vv_f32m2
#define __riscv_vfmul_vf_f32 __riscv_vfmul_vf_f32m2
#define __riscv_vfmul_vv_f32 __riscv_vfmul_vv_f32m2
#define __riscv_vfmv_f_s_f32_f32 __riscv_vfmv_f_s_f32m2_f32
#define __riscv_vfmv_s_f_f32 __riscv_vfmv_s_f_f32m2
#define __riscv_vfmv_v_f_f32 __riscv_vfmv_v_f_f32m2
#define __riscv_vfneg_v_f32 __riscv_vfneg_v_f32m2
#define __riscv_vfredmax_vs_f32_f32 __riscv_vfredmax_vs_f32m2_f32m1
#define __riscv_vfredmin_vs_f32_f32 __riscv_vfredmin_vs_f32m2_f32m1
#define __riscv_vfredusum_vs_f32_f32 __riscv_vfredusum_vs_f32m2_f32m1
#define __riscv_vfredosum_vs_f32_f32 __riscv_vfredosum_vs_f32m2_f32m1
#define __riscv_vfsub_vv_f32 __riscv_vfsub_vv_f32m2
#define __riscv_vle32_v_f32 __riscv_vle32_v_f32m2
#define __riscv_vlse32_v_f32 __riscv_vlse32_v_f32m2
#define __riscv_vse32_v_f32 __riscv_vse32_v_f32m2
#define __riscv_vsetvl_e32 __riscv_vsetvl_e32m2
#define __riscv_vsetvlmax_e32 __riscv_vsetvlmax_e32m2
#define vfloat32_t vfloat32m2_t
#elif USE_LMUL == 4
#define __riscv_vfabs_v_f32 __riscv_vfabs_v_f32m4
#define __riscv_vfadd_vv_f32 __riscv_vfadd_vv_f32m4
#define __riscv_vfmacc_vv_f32 __riscv_vfmacc_vv_f32m4
#define __riscv_vfmax_vv_f32 __riscv_vfmax_vv_f32m4
#define __riscv_vfmin_vv_f32 __riscv_vfmin_vv_f32m4
#define __riscv_vfmul_vf_f32 __riscv_vfmul_vf_f32m4
#define __riscv_vfmul_vv_f32 __riscv_vfmul_vv_f32m4
#define __riscv_vfmv_f_s_f32_f32 __riscv_vfmv_f_s_f32m4_f32
#define __riscv_vfmv_s_f_f32 __riscv_vfmv_s_f_f32m4
#define __riscv_vfmv_v_f_f32 __riscv_vfmv_v_f_f32m4
#define __riscv_vfneg_v_f32 __riscv_vfneg_v_f32m4
#define __riscv_vfredmax_vs_f32_f32 __riscv_vfredmax_vs_f32m4_f32m1
#define __riscv_vfredmin_vs_f32_f32 __riscv_vfredmin_vs_f32m4_f32m1
#define __riscv_vfredusum_vs_f32_f32 __riscv_vfredusum_vs_f32m4_f32m1
#define __riscv_vfredosum_vs_f32_f32 __riscv_vfredosum_vs_f32m4_f32m1
#define __riscv_vfsub_vv_f32 __riscv_vfsub_vv_f32m4
#define __riscv_vle32_v_f32 __riscv_vle32_v_f32m4
#define __riscv_vlse32_v_f32 __riscv_vlse32_v_f32m4
#define __riscv_vse32_v_f32 __riscv_vse32_v_f32m4
#define __riscv_vsetvl_e32 __riscv_vsetvl_e32m4
#define __riscv_vsetvlmax_e32 __riscv_vsetvlmax_e32m4
#define vfloat32_t vfloat32m4_t
#elif USE_LMUL == 8
#define __riscv_vfabs_v_f32 __riscv_vfabs_v_f32m8
#define __riscv_vfadd_vv_f32 __riscv_vfadd_vv_f32m8
#define __riscv_vfmacc_vv_f32 __riscv_vfmacc_vv_f32m8
#define __riscv_vfmax_vv_f32 __riscv_vfmax_vv_f32m8
#define __riscv_vfmin_vv_f32 __riscv_vfmin_vv_f32m8
#define __riscv_vfmul_vf_f32 __riscv_vfmul_vf_f32m8
#define __riscv_vfmul_vv_f32 __riscv_vfmul_vv_f32m8
#define __riscv_vfmv_f_s_f32_f32 __riscv_vfmv_f_s_f32m8_f32
#define __riscv_vfmv_s_f_f32 __riscv_vfmv_s_f_f32m8
#define __riscv_vfmv_v_f_f32 __riscv_vfmv_v_f_f32m8
#define __riscv_vfneg_v_f32 __riscv_vfneg_v_f32m8
#define __riscv_vfredmax_vs_f32_f32 __riscv_vfredmax_vs_f32m8_f32m1
#define __riscv_vfredmin_vs_f32_f32 __riscv_vfredmin_vs_f32m8_f32m1
#define __riscv_vfredusum_vs_f32_f32 __riscv_vfredusum_vs_f32m8_f32m1
#define __riscv_vfredosum_vs_f32_f32 __riscv_vfredosum_vs_f32m8_f32m1
#define __riscv_vfsub_vv_f32 __riscv_vfsub_vv_f32m8
#define __riscv_vle32_v_f32 __riscv_vle32_v_f32m8
#define __riscv_vlse32_v_f32 __riscv_vlse32_v_f32m8
#define __riscv_vse32_v_f32 __riscv_vse32_v_f32m8
#define __riscv_vsetvl_e32 __riscv_vsetvl_e32m8
#define __riscv_vsetvlmax_e32 __riscv_vsetvlmax_e32m8
#define vfloat32_t vfloat32m8_t
#endif

extern "C"
{

// RVV-specific function declarations
float maxcoeff_rvv(float **a, int n, int m);
float mincoeff_rvv(float **a, int n, int m);
float matnorm_rvv(float **a, int n, int m);
void matneg_rvv(float **a, float **b, int n, int m);
void cwiseabs_rvv(float **a, float **b, int n, int m);
void cwisemin_rvv(float **a, float **b, float **c, int n, int m);
void cwisemax_rvv(float **a, float **b, float **c, int n, int m);
void cwisemul_rvv(float **a, float **b, float **c, int n, int m);
void matmul_rvv(float **a, float **b, float **c, int n, int m, int o);
void matvec_rvv(float **a, float **b, float **c, int n, int m);
void matvec_transpose_rvv(float **a, float **b, float **c, int n, int m);
void matmulf_rvv(float **a, float **b, float f, int n, int m);
void matsub_rvv(float **a, float **b, float **c, int n, int m);
void matadd_rvv(float **a, float **b, float **c, int n, int m);
void transpose_rvv(float **a, float **b, int n, int m);
void matcopy_rvv(float **a, float **b, int n, int m);
void matset_rvv(float **a, float f, int n, int m);
void matsetv_rvv(float **a, float *f, int n, int m);

// Golden (reference) function declarations
float maxcoeff_golden(float **a, int n, int m);
float mincoeff_golden(float **a, int n, int m);
float matnorm_golden(float **a, int n, int m);
void matneg_golden(float **a, float **b, int n, int m);
void cwiseabs_golden(float **a, float **b, int n, int m);
void cwisemin_golden(float **a, float **b, float **c, int n, int m);
void cwisemax_golden(float **a, float **b, float **c, int n, int m);
void cwisemul_golden(float **a, float **b, float **c, int n, int m);
void matmul_golden(float **a, float **b, float **c, int n, int m, int o);
void matvec_golden(float **a, float **b, float **c, int n, int m);
void matvec_transpose_golden(float **a, float **b, float **c, int n, int m);
void matmulf_golden(float **a, float **b, float f, int n, int m);
void matsub_golden(float **a, float **b, float **c, int n, int m);
void matadd_golden(float **a, float **b, float **c, int n, int m);
void transpose_golden(float **a, float **b, int n, int m);
void matcopy_golden(float **a, float **b, int n, int m);
void matset_golden(float **a, float f, int n, int m);
void matsetv_golden(float **a, float *f, int n, int m);

#ifdef USE_RVV
#define maxcoeff maxcoeff_rvv
#define mincoeff mincoeff_rvv
#define matnorm matnorm_rvv
#define matneg matneg_rvv
#define cwiseabs cwiseabs_rvv
#define cwisemin cwisemin_rvv
#define cwisemax cwisemax_rvv
#define cwisemul cwisemul_rvv
#define matmul matmul_rvv
#define matvec matvec_rvv
#define matvec_transpose matvec_transpose_rvv
#define matmulf matmulf_rvv
#define matsub matsub_rvv
#define matadd matadd_rvv
#define transpose transpose_rvv
#define matcopy matcopy_rvv
#define matset matset_rvv
#define matsetv matsetv_rvv
#else
#define maxcoeff maxcoeff_golden
#define mincoeff mincoeff_golden
#define matnorm matnorm_golden
#define matneg matneg_golden
#define cwiseabs cwiseabs_golden
#define cwisemin cwisemin_golden
#define cwisemax cwisemax_golden
#define cwisemul cwisemul_golden
#define matmul matmul_golden
#define matvec matvec_golden
#define matvec_transpose matvec_transpose_golden
#define matmulf matmulf_golden
#define matsub matsub_golden
#define matadd matadd_golden
#define transpose transpose_golden
#define matcopy matcopy_golden
#define matset matset_golden
#define matsetv matsetv_golden
#endif

}
#endif //TINYMPC_RVV_MATLIB_H
