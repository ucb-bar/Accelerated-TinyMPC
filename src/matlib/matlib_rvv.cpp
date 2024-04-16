#include <cstdio>
#include <climits>
#include <cmath>

#include "riscv_vector.h"
#include "matlib/common.h"
#include "matlib/matlib_rvv.h"

extern "C" {

// matrix maximum coefficient
float maxcoeff_golden(float **a, int n, int m) {
    float max = std::numeric_limits<float>::min();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            max = a[i][j] > max ? a[i][j] : max;
        }
    }
    return max;
}

float maxcoeff_rvv(float **a, int n, int m) {
    float max = std::numeric_limits<float>::min();
    vfloat32m1_t vec_max = __riscv_vfmv_s_f_f32m1(max, 1);
    float *ptr_a = &a[0][0];
    int k = m * n;
    for (size_t vl; k > 0; k -= vl, ptr_a += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        vec_max = __riscv_vfredmax_vs_f32m4_f32m1(vec_a, vec_max, vl);
    }
    max = __riscv_vfmv_f_s_f32m1_f32(vec_max);
    return max;
}

// matrix min coefficient
float mincoeff_golden(float **a, int n, int m) {
    float min = std::numeric_limits<float>::max();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            min = a[i][j] < min ? a[i][j] : min;
        }
    }
    return min;
}

float mincoeff(float **a, int n, int m) {
    float min = std::numeric_limits<float>::max();
    vfloat32m1_t vec_min = __riscv_vfmv_s_f_f32m1(min, 1);
    float *ptr_a = &a[0][0];
    int k = m * n;
    for (size_t vl; k > 0; k -= vl, ptr_a += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        vec_min = __riscv_vfredmin_vs_f32m4_f32m1(vec_a, vec_min, vl);
    }
    min = __riscv_vfmv_f_s_f32m1_f32(vec_min);
    return min;
}

// matrix unary negative
void matneg_golden(float **a, float **b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i][j] = -a[i][j];
        }
    }
}

void matneg_rvv(float **a, float **b, int n, int m) {
    int k = m * n;
    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        vfloat32m4_t vec_b = __riscv_vfneg_v_f32m4(vec_a, vl);
        __riscv_vse32_v_f32m4(ptr_b, vec_b, vl);
        // print_array_1d(ptr_b, m, "float", "ptr_b");
    }
}

// matrix coefficient-wise abs
void cwiseabs_golden(float **a, float **b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i][j] = fabs(a[i][j]);
        }
    }
}

void cwiseabs_rvv(float **a, float **b, int n, int m) {
    int k = m * n;
    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        vfloat32m4_t vec_b = __riscv_vfabs_v_f32m4(vec_a, vl);
        __riscv_vse32_v_f32m4(ptr_b, vec_b, vl);
        // print_array_1d(ptr_b, m, "float", "ptr_b");
    }
}

// matrix coefficient-wise min
void cwisemin_golden(float **a, float **b, float **c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i][j] = a[i][j] < b[i][j] ? a[i][j] : b[i][j];
        }
    }
}

void cwisemin_rvv(float **a, float **b, float **c, int n, int m) {
    int k = m * n;
    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    float *ptr_c = &c[0][0];
    for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl, ptr_c += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        vfloat32m4_t vec_b = __riscv_vle32_v_f32m4(ptr_b, vl);
        vfloat32m4_t vec_c = __riscv_vfmin_vv_f32m4(vec_a, vec_b, vl);
        __riscv_vse32_v_f32m4(ptr_c, vec_c, vl);
    }
}
    
// matrix coefficient-wise multiplication
void cwisemul_golden(float **a, float **b, float **c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i][j] = a[i][j] * b[i][j];
        }
    }
}

void cwisemul_rvv(float **a, float **b, float **c, int n, int m) {
    int k = m * n;
    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    float *ptr_c = &c[0][0];
    for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl, ptr_c += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        vfloat32m4_t vec_b = __riscv_vle32_v_f32m4(ptr_b, vl);
        vfloat32m4_t vec_c = __riscv_vfmul_vv_f32m4(vec_a, vec_b, vl);
        __riscv_vse32_v_f32m4(ptr_c, vec_c, vl);
    }
}

// matrix coefficient-wise max
void cwisemax_golden(float **a, float **b, float **c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i][j] = a[i][j] > b[i][j] ? a[i][j] : b[i][j];
        }
    }
}

void cwisemax_rvv(float **a, float **b, float **c, int n, int m) {
    int k = m * n;
    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    float *ptr_c = &c[0][0];
    for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl, ptr_c += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        vfloat32m4_t vec_b = __riscv_vle32_v_f32m4(ptr_b, vl);
        vfloat32m4_t vec_c = __riscv_vfmax_vv_f32m4(vec_a, vec_b, vl);
        __riscv_vse32_v_f32m4(ptr_c, vec_c, vl);
    }
}

// matrix multiplication, note B is not [o][m]
// A[n][o], B[m][o] --> C[n][m];
void matmul_golden(float **a, float **b, float **c, int n, int m, int o) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            c[i][j] = 0;
            for (int k = 0; k < o; ++k)
                c[i][j] += a[i][k] * b[j][k];
        }
}

void matmul_rvv(float **a, float **b, float **c, int n, int m, int o) {
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float *ptr_a = &a[i][0]; // row major
            float *ptr_b = &b[j][0]; // column major
            int k = o;
            vfloat32m4_t vec_s = __riscv_vfmv_v_f_f32m4(0, vlmax);
            for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
                vl = __riscv_vsetvl_e32m4(k);
                // printf("%d %d %d\n", i, j, k);
                // print_array_1d(ptr_a, vl, "float", "ptr_a");
                // print_array_1d(ptr_b, vl, "float", "ptr_b");
                vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
                vfloat32m4_t vec_b = __riscv_vle32_v_f32m4(ptr_b, vl);
                vec_s = __riscv_vfmacc_vv_f32m4(vec_s, vec_a, vec_b, vl);
            }
            vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m4_f32m1(vec_s, vec_zero, vlmax);
            float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
            c[i][j] = sum;
        }
    }
}

/*  a is row major
 *        j
 *    1 2 3 4     9
 *  i 5 6 7 8  *  6
 *    9 8 7 6     5 j
 *                4
 */
void matvec_golden(float **a, float **b, float **c, int n, int m) {
    float *ptr_b = &b[0][0];
    float *ptr_c = &c[0][0];
    for (int i = 0; i < n; ++i) {
        ptr_c[i] = 0;
        for (int j = 0; j < m; ++j) {
            ptr_c[i] += a[i][j] * ptr_b[j];
        }
    }
}

void matvec_rvv(float **a, float **b, float **c, int n, int m) {
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
    float *ptr_a = &a[0][0]; // row major
    for (int i = 0; i < n; ++i) {
        int k = m;
        float *ptr_b = &b[0][0];
        vfloat32m4_t vec_s = __riscv_vfmv_v_f_f32m4(0, vlmax);
        for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
            vl = __riscv_vsetvl_e32m4(k);
            vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
            vfloat32m4_t vec_b = __riscv_vle32_v_f32m4(ptr_b, vl);
            vec_s = __riscv_vfmacc_vv_f32m4(vec_s, vec_a, vec_b, vl);
        }
        vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m4_f32m1(vec_s, vec_zero, vlmax);
        float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
        c[i][0] = sum;
    }
}


/*  a is col major
 *      j         i
 *  9 6 5 4  *  1 5 9
 *              2 6 8
 *              3 7 7 j
 *              4 8 6
 */
void matvec_transpose_golden(float **a, float **b, float **c, int n, int m) {
    float *ptr_b = &b[0][0];
    float *ptr_c = &c[0][0];
    for (int i = 0; i < m; ++i) {
        ptr_c[i] = 0;
        for (int j = 0; j < n; ++j) {
            ptr_c[i] += a[j][i] * ptr_b[j];
        }
    }
}

void matvec_transpose_rvv(float **a, float **b, float **c, int n, int m) {
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int i = 0; i < m; ++i) {
        int k = n;
        float *ptr_a = &a[0][i]; // column major
        float *ptr_b = &b[0][0];
        vfloat32m4_t vec_s = __riscv_vfmv_v_f_f32m4(0, vlmax);
        for (size_t vl; k > 0; k -= vl, ptr_a += m * vl, ptr_b += vl) {
            vl = __riscv_vsetvl_e32m4(k);
            vfloat32m4_t vec_a = __riscv_vlse32_v_f32m4(ptr_a, m * sizeof(float), vl);
            vfloat32m4_t vec_b = __riscv_vle32_v_f32m4(ptr_b, vl);
            vec_s = __riscv_vfmacc_vv_f32m4(vec_s, vec_a, vec_b, vl);
        }
        vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m4_f32m1(vec_s, vec_zero, vlmax);
        float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
        c[0][i] = sum;
    }
}

// matrix scalar multiplication
void matmulf_golden(float **a, float **b, float f, int n, int m) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            b[i][j] = f * a[i][j];
        }
}

void matmulf_rvv(float **a, float **b, float f, int n, int m) {
    int k = m * n;
    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        vfloat32m4_t vec_b = __riscv_vfmul_vf_f32m4(vec_a, f, vl);
        __riscv_vse32_v_f32m4(ptr_b, vec_b, vl);
    }
}

// matrix subtraction
void matsub_golden(float **a, float **b, float **c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
}

void matsub_rvv(float **a, float **b, float **c, int n, int m) {
    int k = m * n;
    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    float *ptr_c = &c[0][0];
    for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl, ptr_c += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        vfloat32m4_t vec_b = __riscv_vle32_v_f32m4(ptr_b, vl);
        vfloat32m4_t vec_c = __riscv_vfsub_vv_f32m4(vec_a, vec_b, vl);
        __riscv_vse32_v_f32m4(ptr_c, vec_c, vl);
    }
}

// matrix addition
void matadd_golden(float **a, float **b, float **c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

void matadd_rvv(float **a, float **b, float **c, int n, int m) {
    int k = m * n;
    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    float *ptr_c = &c[0][0];
    for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl, ptr_c += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        vfloat32m4_t vec_b = __riscv_vle32_v_f32m4(ptr_b, vl);
        vfloat32m4_t vec_c = __riscv_vfadd_vv_f32m4(vec_a, vec_b, vl);
        __riscv_vse32_v_f32m4(ptr_c, vec_c, vl);
    }
}

// matrix transpose
void transpose_golden(float **a, float **b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[j][i] = a[i][j];
        }
    }
}

void transpose_rvv(float **a, float **b, int n, int m) {
    for (int j = 0; j < m; ++j) {
        float *ptr_a = &a[0][j];
        float *ptr_b = &b[j][0];
        int k = n;
        int l = 0;
        // for (size_t vl; k > 0; k -= vl, ptr_a += vl * m, ptr_b += vl) {
        for (size_t vl; k > 0; k -= vl, l += vl, ptr_a = &a[l][j], ptr_b += vl) {
            vl = __riscv_vsetvl_e32m4(k);
            vfloat32m4_t vec_a = __riscv_vlse32_v_f32m4(ptr_a, sizeof(float) * m, vl);
            // for (int q = 0; q < vl; q++) {
            //    print_array_1d(&a[l+q][j], 1, "float", "ptr_a");
            // }
            __riscv_vse32(ptr_b, vec_a, vl);
            // print_array_1d(ptr_b, vl, "float", "ptr_b");
        }
    }
};

// matrix copy
void matcopy_golden(float **a, float **b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i][j] = a[i][j];
        }
    }
}

void matcopy_rvv(float **a, float **b, int n, int m) {
    int k = n * m;
    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        __riscv_vse32_v_f32m4(ptr_b, vec_a, vl);
    }
}

void matset_golden(float **a, float f, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            a[i][j] = f;
        }
    }
}

void matset_rvv(float **a, float f, int n, int m) {
    float *ptr_a = &a[0][0];
    int k = m * n;
    for (size_t vl; k > 0; k -= vl, ptr_a += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vfmv_v_f_f32m4(f, vl);
        __riscv_vse32_v_f32m4(ptr_a, vec_a, vl);
    }
}

void matsetv_golden(float **a, float *f, int n, int m) {
    for (int i = 0, k = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j, ++k) {
            a[i][j] = f[k];
        }
    }
}

void matsetv_rvv(float **a, float *f, int n, int m) {
    int k = m * n;
    float *ptr_f = f;
    float *ptr_a = &a[0][0];
    for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_f += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_f = __riscv_vle32_v_f32m4(ptr_f, vl);;
        __riscv_vse32_v_f32m4(ptr_a, vec_f, vl);
    }
}

// matrix l2 norm
float matnorm_golden(float **a, int n, int m) {
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            sum += a[i][j] * a[i][j];
        }
    }
    return sqrt(sum);
}

float matnorm_rvv(float **a, int n, int m) {
    int k = m * n;
    float *ptr_a = &a[0][0];
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m4_t vec_s = __riscv_vfmv_v_f_f32m4(0, vlmax);
    for (size_t vl; k > 0; k -= vl, ptr_a += vl) {
        vl = __riscv_vsetvl_e32m4(k);
        vfloat32m4_t vec_a = __riscv_vle32_v_f32m4(ptr_a, vl);
        vec_s = __riscv_vfmacc_vv_f32m4(vec_s, vec_a, vec_a, vl);
    }
    vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m4_f32m1(vec_s, vec_zero, vlmax);
    float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    return sqrt(sum);
}

}
