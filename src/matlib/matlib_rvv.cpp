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

float maxcoeff(float **a, int n, int m) {
    float max = std::numeric_limits<float>::min();
    vfloat32m1_t vec_max = __riscv_vfmv_s_f_f32m1(max, 1);
    for (int i = 0; i < n; ++i) {
        float *ptr_a = &a[i][0];
        int k = m;
        for (size_t vl; k > 0; k -= vl, ptr_a += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
            vec_max = __riscv_vfredmax_vs_f32m1_f32m1(vec_a, vec_max, vl);
        }
    }
    max = __riscv_vfmv_f_s_f32m1_f32(vec_max);
    return max;
}

// matrix unary negative
void matneg_golden(float **a, float **b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i][j] = -a[i][j];
        }
    }
}

void matneg(float **a, float **b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        float *ptr_a = &a[i][0];
        float *ptr_b = &b[i][0];
        int k = m;
        for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
            vfloat32m1_t vec_b = __riscv_vfneg_v_f32m1(vec_a, vl);
            __riscv_vse32_v_f32m1(ptr_b, vec_b, vl);
            // print_array_1d(ptr_b, m, "float", "ptr_b");
        }
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

void cwiseabs(float **a, float **b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        float *ptr_a = &a[i][0];
        float *ptr_b = &b[i][0];
        int k = m;
        for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
            vfloat32m1_t vec_b = __riscv_vfabs_v_f32m1(vec_a, vl);
            __riscv_vse32_v_f32m1(ptr_b, vec_b, vl);
            // print_array_1d(ptr_b, m, "float", "ptr_b");
        }
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

void cwisemin(float **a, float **b, float **c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        int k = m;
        float *ptr_a = &a[i][0];
        float *ptr_b = &b[i][0];
        float *ptr_c = &c[i][0];
        for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl, ptr_c += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(ptr_b, vl);
            vfloat32m1_t vec_c = __riscv_vfmin_vv_f32m1(vec_a, vec_b, vl);
            __riscv_vse32_v_f32m1(ptr_c, vec_c, vl);
        }
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

void cwisemul(float **a, float **b, float **c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        int k = m;
        float *ptr_a = &a[i][0];
        float *ptr_b = &b[i][0];
        float *ptr_c = &c[i][0];
        for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl, ptr_c += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(ptr_b, vl);
            vfloat32m1_t vec_c = __riscv_vfmul_vv_f32m1(vec_a, vec_b, vl);
            __riscv_vse32_v_f32m1(ptr_c, vec_c, vl);
        }
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

void cwisemax(float **a, float **b, float **c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        int k = m;
        float *ptr_a = &a[i][0];
        float *ptr_b = &b[i][0];
        float *ptr_c = &c[i][0];
        for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl, ptr_c += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(ptr_b, vl);
            vfloat32m1_t vec_c = __riscv_vfmax_vv_f32m1(vec_a, vec_b, vl);
            __riscv_vse32_v_f32m1(ptr_c, vec_c, vl);
        }
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

void matmul(float **a, float **b, float **c, int n, int m, int o) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float *ptr_a = &a[i][0]; // row major
            float *ptr_b = &b[j][0]; // column major
            int k = o;
            vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);
            for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
                vl = __riscv_vsetvl_e32m1(k);
                // printf("%d %d %d\n", i, j, k);
                // print_array_1d(ptr_a, vl, "float", "ptr_a");
                // print_array_1d(ptr_b, vl, "float", "ptr_b");
                vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
                vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(ptr_b, vl);
                vec_s = __riscv_vfmacc_vv_f32m1(vec_s, vec_a, vec_b, vl);
            }
            vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, vlmax);
            float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
            c[i][j] = sum;
        }
    }
}

// matrix scalar multiplication
void matmulf_golden(float **a, float **b, float f, int n, int m) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            b[i][j] = f * a[i][j];
        }
}

void matmulf(float **a, float **b, float f, int n, int m) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    for (int i = 0; i < n; ++i) {
        float *ptr_a = &a[i][0];
        float *ptr_b = &b[i][0];
        int k = m;
        for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
            vfloat32m1_t vec_b = __riscv_vfmul_vf_f32m1(vec_a, f, vl);
            __riscv_vse32_v_f32m1(ptr_b, vec_b, vl);
        }
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

void matsub(float **a, float **b, float **c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        int k = m;
        float *ptr_a = &a[i][0];
        float *ptr_b = &b[i][0];
        float *ptr_c = &c[i][0];
        for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl, ptr_c += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(ptr_b, vl);
            vfloat32m1_t vec_c = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
            __riscv_vse32_v_f32m1(ptr_c, vec_c, vl);
        }
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

void matadd(float **a, float **b, float **c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        int k = m;
        float *ptr_a = &a[i][0];
        float *ptr_b = &b[i][0];
        float *ptr_c = &c[i][0];
        for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl, ptr_c += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            // printf("%d %d\n", i, k);
            // print_array_1d(ptr_a, vl, "float", "ptr_a");
            // print_array_1d(ptr_b, vl, "float", "ptr_b");
            // print_array_1d(ptr_c, vl, "float", "ptr_c");
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(ptr_b, vl);
            vfloat32m1_t vec_c = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
            __riscv_vse32_v_f32m1(ptr_c, vec_c, vl);
        }
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

void transpose(float **a, float **b, int n, int m) {
    for (int j = 0; j < m; ++j) {
        float *ptr_a = &a[0][j];
        float *ptr_b = &b[j][0];
        int k = n;
        int l = 0;
        // for (size_t vl; k > 0; k -= vl, ptr_a += vl * m, ptr_b += vl) {
        for (size_t vl; k > 0; k -= vl, l += vl, ptr_a = &a[l][j], ptr_b += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            vfloat32m1_t vec_a = __riscv_vlse32_v_f32m1(ptr_a, sizeof(float) * m, vl);
            for (int q = 0; q < vl; q++) {
                // print_array_1d(&a[l+q][j], 1, "float", "ptr_a");
            }
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

void matcopy(float **a, float **b, int n, int m) {
    float *ptr_a = &a[0][0];
    float *ptr_b = &b[0][0];
    int k = n * m;
    for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
        vl = __riscv_vsetvl_e32m1(k);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
        __riscv_vse32_v_f32m1(ptr_b, vec_a, vl);
    }
}

void matset_golden(float **a, float f, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            a[i][j] = f;
        }
    }
}

void matset(float **a, float f, int n, int m) {
    for (int i = 0; i < n; ++i) {
        float *ptr_a = &a[i][0];
        int k = m;
        for (size_t vl; k > 0; k -= vl, ptr_a += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            vfloat32m1_t vec_a = __riscv_vfmv_v_f_f32m1(f, vl);
            __riscv_vse32_v_f32m1(ptr_a, vec_a, vl);
        }
    }
}

void matsetv_golden(float **a, float *f, int n, int m) {
    for (int i = 0, k = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j, ++k) {
            a[i][j] = f[k];
        }
    }
}

void matsetv(float **a, float *f, int n, int m) {
    float *ptr_f = f;
    for (int i = 0; i < n; ++i) {
        float *ptr_a = &a[i][0];
        int j = m;
        for (size_t vl; j > 0; j -= vl, ptr_a += vl, ptr_f += vl) {
            vl = __riscv_vsetvl_e32m1(j);
            vfloat32m1_t vec_f = __riscv_vle32_v_f32m1(ptr_f, vl);;
            __riscv_vse32_v_f32m1(ptr_a, vec_f, vl);
        }
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

float matnorm(float **a, int n, int m) {
    float sum = 0;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
    for (int i = 0; i < n; ++i) {
        float *ptr_a = &a[i][0];
        int k = m;
        vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);
        for (size_t vl; k > 0; k -= vl, ptr_a += vl) {
            vl = __riscv_vsetvl_e32m1(k);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(ptr_a, vl);
            vec_s = __riscv_vfmacc_vv_f32m1(vec_s, vec_a, vec_a, vl);
        }
        vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, vlmax);
        sum += __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    }
    return sqrt(sum);
}

}
