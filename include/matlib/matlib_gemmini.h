#pragma once
#ifndef TINYMPC_MATLIB_GEMMINI_H
#define TINYMPC_MATLIB_GEMMINI_H

#include <cmath>
#include "gemmini.h"

extern "C"
{

// Golden (reference) function declarations
float maxcoeff_gemmini(float *a, int n, int m);
float mincoeff_gemmini(float *a, int n, int m);
float matnorm_gemmini(float *a, int n, int m);
void matneg_gemmini(float *a, float *b, int n, int m);
void cwiseabs_gemmini(float *a, float *b, int n, int m);
void cwisemin_gemmini(float *a, float *b, float *c, int n, int mr);
void cwisemax_gemmini(float *a, float *b, float *c, int n, int m);
void cwisemul_gemmini(float *a, float *b, float *c, int n, int m);
void matmul_gemmini(float *a, float *b, float *c, int n, int m, int o);
void matvec_gemmini(float *a, float *b, float *c, int n, int m);
void matvec_transpose_gemmini(float *a, float *b, float *c, int n, int m);
void matmulf_gemmini(float *a, float *b, float f, int n, int m);
void matsub_gemmini(float *a, float *b, float *c, int n, int m);
void matadd_gemmini(float *a, float *b, float *c, int n, int m);
void transpose_gemmini(float *a, float *b, int n, int m);
void matcopy_gemmini(const float *a, float *b, int n, int m);
void matset_gemmini(float *a, float f, int n, int m);
void matsetv_gemmini(float *a, float *f, int n, int m);

#ifdef USE_GEMMINI
#define maxcoeff maxcoeff_gemmini
#define mincoeff mincoeff_gemmini
#define matnorm matnorm_gemmini
#define matneg matneg_gemmini
#define cwiseabs cwiseabs_gemmini
#define cwisemin cwisemin_gemmini
#define cwisemax cwisemax_gemmini
#define cwisemul cwisemul_gemmini
#define matmul matmul_gemmini
#define matvec matvec_gemmini
#define matvec_transpose matvec_transpose_gemmini
#define matmulf matmulf_gemmini
#define matsub matsub_gemmini
#define matadd matadd_gemmini
#define transpose transpose_gemmini
#define matcopy matcopy_gemmini
#define matset matset_gemmini
#define matsetv matsetv_gemmini
#endif

}

// matrix maximum coefficient
inline float maxcoeff_gemmini(float *a, int n, int m) {
    float max = std::numeric_limits<float>::min();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            max = a[i * m + j] > max ? a[i * m + j] : max;
        }
    }
    return max;
}

// matrix min coefficient
inline float mincoeff_gemmini(float *a, int n, int m) {
    float min = std::numeric_limits<float>::max();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            min = a[i * m + j] < min ? a[i * m + j] : min;
        }
    }
    return min;
}

// matrix unary negative
inline void matneg_gemmini(float *a, float *b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i * m + j] = -a[i * m + j];
        }
    }
}

// matrix coefficient-wise abs
inline void cwiseabs_gemmini(float *a, float *b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i * m + j] = fabs(a[i * m + j]);
        }
    }
}

// matrix coefficient-wise min
inline void cwisemin_gemmini(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = a[i * m + j] < b[i * m + j] ? a[i * m + j] : b[i * m + j];
        }
    }
}

// matrix coefficient-wise multiplication
inline void cwisemul_gemmini(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = a[i * m + j] * b[i * m + j];
        }
    }
}

// matrix coefficient-wise max
inline void cwisemax_gemmini(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = a[i * m + j] > b[i * m + j] ? a[i * m + j] : b[i * m + j];
        }
    }
}

    // void tiled_matmul_outer_eigen (
    //     const Matrix<float, Dynamic, Dynamic, RowMajor>&A,
    //     const Matrix<float, Dynamic, Dynamic, RowMajor>&B,
    //     Matrix<float, Dynamic, Dynamic, RowMajor>&C,
    //     bool transpose_A, bool transpose_B) 
    // {
    //         int i = transpose_A ? A.cols() : A.rows();
    //         int j = transpose_B ? B.rows() : B.cols();
    //         int k = transpose_B ? B.cols() : B.rows();
    //         int tile_I = (i + DIM - 1) / DIM;
    //         int tile_J = (j + DIM - 1) / DIM;
    //         int tile_K = (k + DIM - 1) / DIM;
    //         tiled_matmul_outer_simple(i, j, k,
    //                 A.data(), B.data(), NULL, C.data(),
    //                 transpose_A ? i : k, transpose_B ? k : j, j, j,
    //                 MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
    //                 tile_I, tile_J, tile_K,
    //                 NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
    //                 transpose_A, transpose_B,
    //                 false, false,
    //                 0,
    //                 WS
    //                 );
    // }

// inline void matmul_gemmini(float *a, float *b, float *c, int n, int m, int o) {
//     int tile_N = (n + DIM - 1) / DIM;
//     int tile_M = (m + DIM - 1) / DIM;
//     int tile_O = (o + DIM - 1) / DIM;

//     tiled_matmul_outer_simple(
//         n, m, o,                  // Dimensions of matrices
//         a, b, NULL, c,            // Pointers to matrix data
//         o, o, m, m,               // Leading dimensions
//         MVIN_SCALE_IDENTITY,      // Input scaling for matrix A
//         MVIN_SCALE_IDENTITY,      // Input scaling for matrix B
//         MVIN_SCALE_IDENTITY,      // Input scaling for matrix C
//         tile_N, tile_M, tile_O,   // Tile sizes
//         NO_ACTIVATION,            // Activation function
//         ACC_SCALE_IDENTITY,       // Accumulator scaling
//         0,                        // Bias
//         false,                    // ReLU after operations
//         false, false,             // Transpose flags for A and B
//         false, false,             // Flush flags for A and B
//         0,                        // Exponent Bias
//         WS                        // Scratchpad
//     );
// }

// matrix multiplication, note B is not [o][m]
// A[n][o], B[m][o] --> C[n][m];
inline void matmul_gemmini(float *a, float *b, float *c, int n, int m, int o) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = 0;
            for (int k = 0; k < o; ++k) {
                // printf("(%0.5f * %0.5f)\n", a[i * o + k], b[j * o + k]);
                // if(k%4 == 0) {
                //     printf("t(%2d) c[%d][%d] = %0.5f\n", k, i, j, c[i * m + j]);
                // }
                c[i * m + j] += a[i * o + k] * b[j * o + k];
                // if(k == o - 1) {
                //     printf("t(%2d) c[%d][%d] = %0.5f\n", k, i, j, c[i * m + j]);
                // }
            }
        }
}
    // int I = n/DIM;
    // int J = m/DIM;
    // int K = o/DIM;

    // const uint32_t A_sp_addr_start = 0;
    // const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
    // const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);

    // size_t A_row_stride = k;
    // size_t B_row_stride = j;

    // for (size_t k = 0; k < K; k++) {
    //     for (size_t j = 0; j < J; j++) {
    //         for (size_t i = 0; i < I; i++) {
    //             gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);

    //         }
    //     }
    // }


//     size_t I = n/DIM;
//     size_t J = m/DIM;
//     size_t K = o/DIM;

//     int pad_I = 0;
//     int pad_J = 3;
//     int pad_K = 0;


//     sp_tiled_matmul_ws(
//             a, b, NULL, c,
//             1.0, 1.0, 1.0
//             I, J, K, pad_I, pad_J, pad_K,
//             o, o, o, m,
//             false, false,
//             true, false,
//             false, false,
//             NO_ACTIVATION,
//             0, 0);

// static inline void sp_tiled_matmul_ws(
//         const elem_t * A, const elem_t * B, const void * D, void * C,
//         scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
//         size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
//         size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
//         bool a_transpose, bool b_transpose,
//         bool full_C, bool low_D,
//         bool no_bias, bool repeating_bias,
//         int act,
//         int a_spad_id, int b_spad_id) {
// }

/*  a is row major
 *        j
 *    1 2 3 4     9
 *  i 5 6 7 8  *  6
 *    9 8 7 6     5 j
 *                4
 */
inline void matvec_gemmini(float *a, float *b, float *c, int n, int m) {
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
inline void matvec_transpose_gemmini(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < m; ++i) {
        c[i] = 0;
        for (int j = 0; j < n; ++j) {
            c[i] += a[j * m + i] * b[j];
        }
    }
}

// matrix scalar multiplication
inline void matmulf_gemmini(float *a, float *b, float f, int n, int m) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            b[i * m + j] = f * a[i * m + j];
        }
}

// matrix subtraction
inline void matsub_gemmini(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = a[i * m + j] - b[i * m + j];
        }
    }
}

// matrix addition
inline void matadd_gemmini(float *a, float *b, float *c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            c[i * m + j] = a[i * m + j] + b[i * m + j];
        }
    }
}

// matrix transpose
inline void transpose_gemmini(float *a, float *b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[j * n + i] = a[i * m + j];
        }
    }
}

// matrix copy
inline void matcopy_gemmini(const float *a, float *b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i * m + j] = a[i * m + j];
        }
    }
}

inline void matset_gemmini(float *a, float f, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            a[i * m + j] = f;
        }
    }
}

inline void matsetv_gemmini(float *a, float *f, int n, int m) {
    for (int i = 0, k = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j, ++k) {
            a[i * m + j] = f[k];
        }
    }
}

// matrix l2 norm
inline float matnorm_gemmini(float *a, int n, int m) {
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            sum += a[i * m + j] * a[i * m + j];
        }
    }
    return sqrt(sum);
}

#endif // TINYMPC_MATLIB_GEMMINI_H
