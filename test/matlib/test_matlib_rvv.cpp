#include <cstdio>
#include <cmath>
#include <cstdint>

#include "matlib/common.h"
#include "matlib/matlib_rvv.h"

extern "C" {

int main() {
    const int N = 11;
    const int M = 13;
    const int O = 7;
    uint32_t seed = 0xdeadbeef;
    srand(seed);

    // data gen
    float **A = alloc_array_2d(N, O);
    float **B = alloc_array_2d(M, O);
    float *f = alloc_array_1d(N * M);
    gen_rand_2d(A, N, O);
    gen_rand_2d(B, M, O);
    gen_rand_1d(f, N * M);

    printf("matmul:         ");
    float **golden = alloc_array_2d(N, M);
    float **actual = alloc_array_2d(N, M);
    // print_array_2d(A, N, O, "float", "A");
    // print_array_2d(B, M, O, "float", "B");
    matmul_golden(A, B, golden, N, M, O);
    matmul(A, B, actual, N, M, O);
    // print_array_2d(golden, N, M, "float", "golden");
    // print_array_2d(actual, N, M, "float", "actual");
    puts(compare_2d(golden, actual, N, M) ? "pass" : "fail");

    // data gen
    float **G = alloc_array_2d(N, M);
    float **H = alloc_array_2d(M, N);
    float **V = alloc_array_2d(1, M);
    float **W = alloc_array_2d(M, 1);
    gen_rand_2d(G, N, M);
    gen_rand_2d(H, M, N);
    gen_rand_2d(V, 1, M);
    gen_rand_2d(W, M, 1);

    printf("matvec:         ");
    float **golden_vec = alloc_array_2d(N, 1);
    float **actual_vec = alloc_array_2d(N, 1);
    matvec_golden(G, V, golden_vec, N, M);
    matvec(G, V, actual_vec, N, M);
    puts(compare_2d(golden_vec, actual_vec, N, 1) ? "pass" : "fail");

    printf("matvec_t:       ");
    // print_array_2d(H, M, N, "float", "H");
    // print_array_2d(W, M, 1, "float", "W");
    matvec_transpose_golden(H, W, golden_vec, M, N);
    matvec_transpose(H, W, actual_vec, M, N);
    // print_array_2d(golden_vec, N, 1, "float", "golden_vec");
    // print_array_2d(actual_vec, N, 1, "float", "actual_vec");
    puts(compare_2d(golden_vec, actual_vec, N, 1) ? "pass" : "fail");

    // data gen
    A = alloc_array_2d(N, M);
    B = alloc_array_2d(N, M);
    float **C = alloc_array_2d(N, M);
    float **D = alloc_array_2d(N, M);
    gen_rand_2d(A, N, M);
    gen_rand_2d(B, N, M);

    printf("maxcoeff:       ");
    float max_golden = maxcoeff_golden(A, N, M);
    float max_actual = maxcoeff(A, N, M);
    puts(float_eq(max_golden, max_actual, 1e-6) ? "pass" : "fail");

    printf("mincoeff:       ");
    float min_golden = mincoeff_golden(A, N, M);
    float min_actual = mincoeff(A, N, M);
    puts(float_eq(min_golden, min_actual, 1e-6) ? "pass" : "fail");

    printf("matmulf:        ");
    matmulf_golden(A, C, 10.0f, N, M);
    matmulf(A, D, 10.0f, N, M);
    puts(compare_2d(C, D, N, M) ? "pass" : "fail");

    printf("matsub:         ");
    matsub_golden(A, B, C, N, M);
    matsub(A, B, D, N, M);
    puts(compare_2d(C, D, N, M) ? "pass" : "fail");

    printf("matadd:         ");
    matadd_golden(A, B, C, N, M);
    matadd(A, B, D, N, M);
    puts(compare_2d(C, D, N, M) ? "pass" : "fail");

    printf("matneg:         ");
    matneg_golden(A, C, N, M);
    matneg(A, D, N, M);
    puts(compare_2d(C, D, N, M) ? "pass" : "fail");

    printf("matcopy:        ");
    matcopy_golden(A, C, N, M);
    matcopy(A, D, N, M);
    puts(compare_2d(C, D, N, M) ? "pass" : "fail");

    printf("cwiseabs:       ");
    cwiseabs_golden(A, C, N, M);
    cwiseabs(A, D, N, M);
    puts(compare_2d(C, D, N, M) ? "pass" : "fail");

    printf("cwisemin:       ");
    cwisemin_golden(A, B, C, N, M);
    cwisemin(A, B, D, N, M);
    puts(compare_2d(C, D, N, M) ? "pass" : "fail");

    printf("cwisemax:       ");
    cwisemax_golden(A, B, C, N, M);
    cwisemax(A, B, D, N, M);
    puts(compare_2d(C, D, N, M) ? "pass" : "fail");

    printf("cwisemul:       ");
    cwisemul_golden(A, B, C, N, M);
    cwisemul(A, B, D, N, M);
    puts(compare_2d(C, D, N, M) ? "pass" : "fail");

    printf("matset:         ");
    matset_golden(A, 5.0, N, M);
    matset(A, 5.0, N, M);
    puts(compare_2d(C, D, N, M) ? "pass" : "fail");

    printf("matsetv:        ");
    matsetv_golden(A, f, N, M);
    matsetv(A, f, N, M);
    puts(compare_2d(C, D, N, M) ? "pass" : "fail");

    printf("matnorm:        ");
    float norm_golden = matnorm_golden(A, N, M);
    float norm_actual = matnorm(A, N, M);
    puts(float_eq(norm_golden, norm_actual, 1e-6) ? "pass" : "fail");

    C = alloc_array_2d(M, N);
    D = alloc_array_2d(M, N);

    printf("transpose:      ");
    transpose_golden(A, C, N, M);
    transpose(A, D, N, M);
    puts(compare_2d(C, D, M, N) ? "pass" : "fail");

done:
    return(0);
}

}
