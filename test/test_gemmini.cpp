#include <iostream>
#include "gemmini.h"
#include <vector>
#include <cstdlib>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

// I4 works, possibly because it can be moved to scratchpad with one mvin instruction
float I4[16] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
};

float I12[144] = {
    1.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	
    0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	
    0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	
    0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	
    0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	
    0.0,	0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	
    0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	0.0,	
    0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	0.0,	
    0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	
    0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	
    0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	1.0,	0.0,	
    0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	1.0
};

float I12_4[144] = {
    1.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	1.0,	0.0,	0.0,	0.0,	
    0.0,	2.0,	0.0,	0.0,	0.0,	2.0,	0.0,	0.0,	0.0,	2.0,	0.0,	0.0,	
    0.0,	0.0,	3.0,	0.0,	0.0,	0.0,	3.0,	0.0,	0.0,	0.0,	3.0,	0.0,	
    0.0,	0.0,	0.0,	4.0,	0.0,	0.0,	0.0,	4.0,	0.0,	0.0,	0.0,	4.0
};

float test_x[4] = {
    1, 2, 3, 4
};

float test_x_12[12] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
};

float result_4[4] = {
    0, 0, 0, 0
};

float result_12[12] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

void tiled_matmul_spad_dram(
        const uint32_t sp_A_addr,
        const Matrix<float, Dynamic, Dynamic, RowMajor>&B,
        Matrix<float, Dynamic, Dynamic, RowMajor>&C,
        int i) {
    int j = B.cols();
    int k = B.rows();
    int tile_I = (i + DIM - 1) / DIM;
    int tile_J = (j + DIM - 1) / DIM;
    int tile_K = (k + DIM - 1) / DIM;

    tiled_matmul_outer_simple_dram_sp(i, j, k,
            sp_A_addr, B.data(), NULL, C.data(),
            k, j, j, j,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            tile_I, tile_J, tile_K,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, false,
            0,
            WS
            );
}

void mul_I4() {
    // mvin I4
    gemmini_extended3_config_ld(16, 1.0, false, 0);
    gemmini_extended_mvin(I4, 0, 4, 4);

    int i = 4;
    int j = 1;
    int k = 4;
    int tile_I = (i + DIM - 1) / DIM;
    int tile_J = (j + DIM - 1) / DIM;
    int tile_K = (k + DIM - 1) / DIM;

    tiled_matmul_outer_simple_dram_sp(i, j, k,
            0, test_x, NULL, result_4,
            k, j, j, j,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            tile_I, tile_J, tile_K,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, false,
            0,
            WS
    );

    printf("result: {%f, %f, %f, %f}\n", result_4[0], result_4[1], result_4[2], result_4[3]);
}

void mul_I12() {
    gemmini_extended3_config_ld(48, 1.0, false, 0);

    // for (int i = 0; i < 3; i++) {
    //     printf("moving in data at iteration %d, starting with element %f\n", i, *(I12 + i*(48*4)));
    //     gemmini_extended_mvin(I12 + i*(48*4), 0 + i*12, 12, 4);
    // }
    gemmini_extended_mvin(I12, 0, 12, 4);
    gemmini_extended_mvin(I12 + 48, 12, 12, 4);
    gemmini_extended_mvin(I12 + 96, 24, 12, 4);

    int i = 12;
    int j = 1;
    int k = 12;
    int tile_I = (i + DIM - 1) / DIM;
    int tile_J = (j + DIM - 1) / DIM;
    int tile_K = (k + DIM - 1) / DIM;

    tiled_matmul_outer_simple_dram_sp(i, j, k,
            0, test_x_12, NULL, result_12,
            k, j, j, j,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            tile_I, tile_J, tile_K,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, false,
            0,
            WS
    );

    printf("result: {%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f}\n", 
                result_12[0], result_12[1], result_12[2], result_12[3],
                result_12[4], result_12[5], result_12[6], result_12[7],
                result_12[8], result_12[9], result_12[10], result_12[11]);
}

void mul_I12_4() {
    // mvin I4
    gemmini_extended3_config_ld(48, 1.0, false, 0);
    gemmini_extended_mvin(I12_4, 0, 12, 4);

    int i = 4;
    int j = 1;
    int k = 12;
    int tile_I = (i + DIM - 1) / DIM;
    int tile_J = (j + DIM - 1) / DIM;
    int tile_K = (k + DIM - 1) / DIM;

    tiled_matmul_outer_simple_dram_sp(i, j, k,
            0, test_x_12, NULL, result_4,
            k, j, j, j,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            tile_I, tile_J, tile_K,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, false,
            0,
            WS
    );

    printf("result: {%f, %f, %f, %f}\n", 
                result_4[0], result_4[1], result_4[2], result_4[3]);
}

int main(int argc, char* argv[]) {
    // mul_I4();
    // mul_I12();
    // mul_I12_4();
    return 0;
}