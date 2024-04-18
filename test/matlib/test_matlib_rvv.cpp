#include <cstdio>
#include <cmath>

#include "tinympc/types_rvv.hpp"

extern "C" {

int main() {

    enable_vector_operations();

    const int N = 11;
    const int M = 13;
    const int O = 7;
    uint32_t seed = 0xdeadbeef;
    uint64_t start, total;
    srand(seed);

    // array gen
    float *A = alloc_array_2d(N, O);
    float *B = alloc_array_2d(M, O);
    float *f = alloc_array_1d(N * M);
    gen_rand_2d(A, N, O);
    gen_rand_2d(B, M, O);
    gen_rand_1d(f, N * M);

    printf("matmul:         ");
    float *golden = alloc_array_2d(N, M);
    float *actual = alloc_array_2d(N, M);
    matmul_golden(A, B, golden, N, M, O);
    start = read_cycles();
    matmul(A, B, actual, N, M, O);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(golden, actual, N, M) ? "pass" : "fail", total);

    // array gen
    float *G = alloc_array_2d(N, M);
    float *H = alloc_array_2d(M, N);
    float *V = alloc_array_2d(1, M);
    float *W = alloc_array_2d(M, 1);
    gen_rand_2d(G, N, M);
    gen_rand_2d(H, M, N);
    gen_rand_2d(V, 1, M);
    gen_rand_2d(W, M, 1);

    printf("matvec:         ");
    float *golden_vec = alloc_array_2d(N, 1);
    float *actual_vec = alloc_array_2d(N, 1);
    matvec_golden(G, V, golden_vec, N, M);
    start = read_cycles();
    matvec(G, V, actual_vec, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(golden_vec, actual_vec, N, 1) ? "pass" : "fail", total);

    printf("matvec_t:       ");
    matvec_transpose_golden(H, W, golden_vec, M, N);
    start = read_cycles();
    matvec_transpose(H, W, actual_vec, M, N);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(golden_vec, actual_vec, N, 1) ? "pass" : "fail", total);

    // array gen
    A = alloc_array_2d(N, M);
    B = alloc_array_2d(N, M);
    float *C = alloc_array_2d(N, M);
    float *D = alloc_array_2d(N, M);
    gen_rand_2d(A, N, M);
    gen_rand_2d(B, N, M);

    printf("maxcoeff:       ");
    float max_golden = maxcoeff_golden(A, N, M);
    start = read_cycles();
    float max_actual = maxcoeff(A, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", float_eq(max_golden, max_actual, 1e-6) ? "pass" : "fail", total);

    printf("mincoeff:       ");
    float min_golden = mincoeff_golden(A, N, M);
    start = read_cycles();
    float min_actual = mincoeff(A, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", float_eq(min_golden, min_actual, 1e-6) ? "pass" : "fail", total);

    printf("matmulf:        ");
    matmulf_golden(A, C, 10.0f, N, M);
    start = read_cycles();
    matmulf(A, D, 10.0f, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matsub:         ");
    matsub_golden(A, B, C, N, M);
    start = read_cycles();
    matsub(A, B, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matadd:         ");
    matadd_golden(A, B, C, N, M);
    start = read_cycles();
    matadd(A, B, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matneg:         ");
    matneg_golden(A, C, N, M);
    start = read_cycles();
    matneg(A, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matcopy:        ");
    matcopy_golden(A, C, N, M);
    start = read_cycles();
    matcopy(A, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("cwiseabs:       ");
    cwiseabs_golden(A, C, N, M);
    start = read_cycles();
    cwiseabs(A, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("cwisemin:       ");
    cwisemin_golden(A, B, C, N, M);
    start = read_cycles();
    cwisemin(A, B, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("cwisemax:       ");
    cwisemax_golden(A, B, C, N, M);
    start = read_cycles();
    cwisemax(A, B, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("cwisemul:       ");
    cwisemul_golden(A, B, C, N, M);
    start = read_cycles();
    cwisemul(A, B, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matset:         ");
    matset_golden(A, 5.0, N, M);
    start = read_cycles();
    matset(A, 5.0, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matsetv:        ");
    matsetv_golden(A, f, N, M);
    start = read_cycles();
    matsetv(A, f, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matnorm:        ");
    float norm_golden = matnorm_golden(A, N, M);
    start = read_cycles();
    float norm_actual = matnorm(A, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", float_eq(norm_golden, norm_actual, 1e-6) ? "pass" : "fail", total);

    C = alloc_array_2d(M, N);
    D = alloc_array_2d(M, N);

    printf("transpose:      ");
    transpose_golden(A, C, N, M);
    start = read_cycles();
    transpose(A, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, M, N) ? "pass" : "fail", total);

done:
    return(0);
}

}
