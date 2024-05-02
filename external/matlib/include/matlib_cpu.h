#pragma once
#ifndef TINYMPC_MATLIB_CPU_H
#define TINYMPC_MATLIB_CPU_H

#include <cmath>

extern "C"
{

// Golden (reference) function declarations
float maxcoeff_cpu(float *a, int n, int m);
float mincoeff_cpu(float *a, int n, int m);
float matnorm_cpu(float *a, int n, int m);
void matneg_cpu(float *a, float *b, int n, int m);
void cwiseabs_cpu(float *a, float *b, int n, int m);
void cwisemin_cpu(float *a, float *b, float *c, int n, int mr);
void cwisemax_cpu(float *a, float *b, float *c, int n, int m);
void cwisemul_cpu(float *a, float *b, float *c, int n, int m);
void matmul_cpu(float *a, float *b, float *c, int n, int m, int o);
void matvec_cpu(float *a, float *b, float *c, int n, int m);
void matvec_transpose_cpu(float *a, float *b, float *c, int n, int m);
void matmulf_cpu(float *a, float *b, float f, int n, int m);
void matsub_cpu(float *a, float *b, float *c, int n, int m);
void matadd_cpu(float *a, float *b, float *c, int n, int m);
void transpose_cpu(float *a, float *b, int n, int m);
void matcopy_cpu(const float *a, float *b, int n, int m);
void matset_cpu(float *a, float f, int n, int m);
void matsetv_cpu(float *a, float *f, int n, int m);

#ifdef USE_CPU
#define maxcoeff maxcoeff_cpu
#define mincoeff mincoeff_cpu
#define matnorm matnorm_cpu
#define matneg matneg_cpu
#define cwiseabs cwiseabs_cpu
#define cwisemin cwisemin_cpu
#define cwisemax cwisemax_cpu
#define cwisemul cwisemul_cpu
#define matmul matmul_cpu
#define matvec matvec_cpu
#define matvec_transpose matvec_transpose_cpu
#define matmulf matmulf_cpu
#define matsub matsub_cpu
#define matadd matadd_cpu
#define transpose transpose_cpu
#define matcopy matcopy_cpu
#define matset matset_cpu
#define matsetv matsetv_cpu
#endif

}

// matrix maximum coefficient
inline float maxcoeff_cpu(float *a, int n, int m) {
    float max = std::numeric_limits<float>::min();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            max = a[i * m + j] > max ? a[i * m + j] : max;
        }
    }
    return max;
}

// matrix min coefficient
inline float mincoeff_cpu(float *a, int n, int m) {
    float min = std::numeric_limits<float>::max();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            min = a[i * m + j] < min ? a[i * m + j] : min;
        }
    }
    return min;
}

// matrix unary negative
inline void matneg_cpu(float *a, float *b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i * m + j] = -a[i * m + j];
        }
    }
}

// matrix coefficient-wise abs
inline void cwiseabs_cpu(float *a, float *b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i * m + j] = fabs(a[i * m + j]);
        }
    }
}

// matrix coefficient-wise min
inline void cwisemin_cpu(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = a[i * m + j] < b[i * m + j] ? a[i * m + j] : b[i * m + j];
        }
    }
}

// matrix coefficient-wise multiplication
inline void cwisemul_cpu(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = a[i * m + j] * b[i * m + j];
        }
    }
}

// matrix coefficient-wise max
inline void cwisemax_cpu(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = a[i * m + j] > b[i * m + j] ? a[i * m + j] : b[i * m + j];
        }
    }
}

// matrix multiplication, note B is not [o][m]
// A[n][o], B[m][o] --> C[n][m];
inline void matmul_cpu(float *a, float *b, float *c, int n, int m, int o) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = 0;
            for (int k = 0; k < o; ++k)
                c[i * m + j] += a[i * o + k] * b[j * o + k];
        }
}

/*  a is row major
 *        j
 *    1 2 3 4     9
 *  i 5 6 7 8  *  6
 *    9 8 7 6     5 j
 *                4
 */
inline void matvec_cpu(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        c[i] = 0;
        for (int j = 0; j < m; ++j) {
            c[i] += a[i * m + j] * b[j];
        }
    }
}

/*  a is col major
 *      j         i
 *  9 6 5 4  *  1 5 9
 *              2 6 8
 *              3 7 7 j
 *              4 8 6
 */
inline void matvec_transpose_cpu(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < m; ++i) {
        c[i] = 0;
        for (int j = 0; j < n; ++j) {
            c[i] += a[j * m + i] * b[j];
        }
    }
}

// matrix scalar multiplication
inline void matmulf_cpu(float *a, float *b, float f, int n, int m) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            b[i * m + j] = f * a[i * m + j];
        }
}

// matrix subtraction
inline void matsub_cpu(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = a[i * m + j] - b[i * m + j];
        }
    }
}

// matrix addition
inline void matadd_cpu(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = a[i * m + j] + b[i * m + j];
        }
    }
}

// matrix transpose
inline void transpose_cpu(float *a, float *b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[j * n + i] = a[i * m + j];
        }
    }
}

// matrix copy
inline void matcopy_cpu(const float *a, float *b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i * m + j] = a[i * m + j];
        }
    }
}

inline void matset_cpu(float *a, float f, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            a[i * m + j] = f;
        }
    }
}

inline void matsetv_cpu(float *a, float *f, int n, int m) {
    for (int i = 0, k = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j, ++k) {
            a[i * m + j] = f[k];
        }
    }
}

// matrix l2 norm
inline float matnorm_cpu(float *a, int n, int m) {
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            sum += a[i * m + j] * a[i * m + j];
        }
    }
    return sqrt(sum);
}

#endif // TINYMPC_MATLIB_CPU_H
