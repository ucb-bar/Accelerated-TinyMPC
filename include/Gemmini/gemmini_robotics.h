#include "gemmini.h"

static inline void dram_sp_tiled_matmul_ws(
        const uint32_t a_sp_addr, const elem_t * B, const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act,
        int a_spad_id, int b_spad_id) {

  const uint32_t A_sp_addr_start = a_sp_addr;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2) | (full_C << (ADDR_LEN-3));

  // const int A_blocks = a_transpose ? (I <= MAX_BLOCK_LEN ? I : MAX_BLOCK_LEN) :
  //   (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN);
  const int B_blocks = b_transpose ? (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN) :
    (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  const int D_blocks = low_D ? (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN) :
    (J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC);
  const int C_blocks = full_C ? 1 : (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  // Move-in D
  if (D != NULL && !no_bias) {
    for (size_t i = 0; i < I; i++) {
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const void * const D_dram_addr = (int8_t *)D + (bias_row * D_row_stride + j)*DIM*sizeof_D;
        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;
        size_t blocks = j + D_blocks <= J ? D_blocks : J-j;
        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        gemmini_extended_mvin3(D_dram_addr, D_sp_addr_acc, cols, rows);
        #ifdef CODEGEN
        printf("gemmini_extended_mvin3([D] + 0x%x, 0x%x, %d, %d);\n", (bias_row * D_row_stride + j)*DIM, D_sp_addr_acc, cols, rows);
        #endif
      }
    }
  }

  for (size_t j = 0; j < J; j++) {
    for (size_t k = 0; k < K; k++) {
      for (size_t i = 0; i < I; i++) {
        const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
          (A_sp_addr_start + (i*K + k)*DIM);
        const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) :
          (B_sp_addr_start + (k*J + j)*DIM);
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        // Mvin B
        if (b_transpose) {
          if (i == 0 && k % B_blocks == 0) {
            const elem_t * const B_dram_addr = B + (j*B_row_stride + k)*DIM;
            const size_t blocks = k + B_blocks <= K ? B_blocks : K-k;
            const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
            const size_t rows = DIM - (j == J-1 ? pad_J : 0);
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
            #ifdef CODEGEN
            printf("gemmini_extended_mvin2([B] + 0x%x, 0x%x, %d, %d);\n", (j*B_row_stride + k)*DIM, B_sp_addr, cols, rows);
            #endif
          }
        } else {
          if (i == 0 && j % B_blocks == 0) {
            const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
            const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
            const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
            const size_t rows = DIM - (k == K-1 ? pad_K : 0);
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
            #ifdef CODEGEN
            printf("gemmini_extended_mvin2([B] + 0x%x, 0x%x, %d, %d);\n", (k*B_row_stride + j)*DIM, B_sp_addr, cols, rows);
            #endif
          }
        }

        // Compute
        {
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr;

          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it
          int no_bias_new_matrix = no_bias && D != NULL && k == 0;
          if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN-2));
          }

          const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
          const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
          const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
          const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

          gemmini_extended_preload(pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);
          #ifdef CODEGEN
          printf("gemmini_extended_preload(0x%x, 0x%x, %d, %d, %d, %d);\n", pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);
          #endif

          if (i == 0) { // First iteration
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #ifdef CODEGEN
            printf("gemmini_extended_compute_preloaded(0x%x, 0x%x, %d, %d, %d, %d);\n", A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #endif
          } else { // All other iterations
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #ifdef CODEGEN
            printf("gemmini_extended_compute_accumulated(0x%x, 0x%x, %d, %d, %d, %d);\n", A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #endif
          }
        }

        // Move-out C
        if (C != NULL && k == K-1 && (j == J-1 || j % C_blocks == C_blocks-1)) {
          const size_t rounded_j = (j / C_blocks) * C_blocks;

          const uint32_t rounded_C_sp_addr = C_sp_addr_start + (i*J + rounded_j)*DIM;
          void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + rounded_j)*DIM*sizeof_C;

          const size_t blocks = rounded_j + C_blocks <= J ? C_blocks : J-rounded_j;
          const size_t cols = blocks * DIM - (rounded_j + blocks >= J ? pad_J : 0);
          const size_t rows = DIM - (i == I - 1 ? pad_I : 0);

          gemmini_extended_mvout(C_dram_addr, rounded_C_sp_addr, cols, rows);
          #ifdef CODEGEN
          printf("gemmini_extended_mvout([C] + 0x%x, 0x%x, %d, %d);\n", (i*C_row_stride + rounded_j)*DIM, rounded_C_sp_addr, cols, rows);
          #endif
        }
      }
    }
  }

  #ifdef CODEGEN
  printf("gemmini_fence();\n");
  printf("\n");
  #endif
}

static inline void spad_spad_tiled_matmul_ws(
        const uint32_t a_sp_addr, const uint32_t b_sp_addr, const void * D, const uint32_t c_sp_addr,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act,
        int a_spad_id, int b_spad_id) {

  const uint32_t A_sp_addr_start = a_sp_addr;
  const uint32_t B_sp_addr_start = b_sp_addr;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = c_sp_addr;

  // const int A_blocks = a_transpose ? (I <= MAX_BLOCK_LEN ? I : MAX_BLOCK_LEN) :
  //   (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN);
  // const int B_blocks = b_transpose ? (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN) :
  //   (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  const int D_blocks = low_D ? (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN) :
    (J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC);
  // const int C_blocks = full_C ? 1 : (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  // const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  // Move-in D
  if (D != NULL && !no_bias) {
    for (size_t i = 0; i < I; i++) {
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const void * const D_dram_addr = (int8_t *)D + (bias_row * D_row_stride + j)*DIM*sizeof_D;
        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;
        size_t blocks = j + D_blocks <= J ? D_blocks : J-j;
        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        gemmini_extended_mvin3(D_dram_addr, D_sp_addr_acc, cols, rows);
        #ifdef CODEGEN
        printf("gemmini_extended_mvin3([D] + 0x%x, 0x%x, %d, %d);\n", (bias_row * D_row_stride + j)*DIM, D_sp_addr_acc, cols, rows);
        #endif
      }
    }
  }

  for (size_t j = 0; j < J; j++) {
    for (size_t k = 0; k < K; k++) {
      for (size_t i = 0; i < I; i++) {
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr;

          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it
          int no_bias_new_matrix = no_bias && D != NULL && k == 0;
          if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN-2));
          }

          const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
          const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
          const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
          const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

          gemmini_extended_preload(pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);
          #ifdef CODEGEN
          printf("gemmini_extended_preload(0x%x, 0x%x, %d, %d, %d, %d);\n", pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);
          #endif

          if (i == 0) { // First iteration
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #ifdef CODEGEN
            printf("gemmini_extended_compute_preloaded(0x%x, 0x%x, %d, %d, %d, %d);\n", A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #endif
          } else { // All other iterations
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #ifdef CODEGEN
            printf("gemmini_extended_compute_accumulated(0x%x, 0x%x, %d, %d, %d, %d);\n", A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #endif
        }
      }
    }
  }

  #ifdef CODEGEN
  printf("gemmini_fence();\n");
  printf("\n");
  #endif
}

static void tiled_matmul_outer_simple_dram_sp(size_t dim_I, size_t dim_J, size_t dim_K,
        const uint32_t sp_A_addr, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA,
        int dataflow) {

  // printf("dim_I: %d\tdim_J: %d\tdim_K: %d\ttile_I: %d\ttile_J: %d\ttile_K: %d\n",
          //  dim_I, dim_J, dim_K, tile_I, tile_J, tile_K);
  const size_t dim_I_padded = tile_I * DIM;
  const size_t dim_J_padded = tile_J * DIM;
  const size_t dim_K_padded = tile_K * DIM;

  // const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
  // const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  // const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);

  const size_t I0 = 1;
  const size_t J0 = 1;
  const size_t K0 = 1;

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I*DIM) == 0 ? tile_I : (dim_I_padded/DIM) % tile_I;
  const size_t last_J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I;
  const size_t padding_J = dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias) {
    D = (void*) 1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose, b_transpose);
  gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
  // gemmini_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
  gemmini_extended3_config_ld(stride_B * sizeof(elem_t), B_scale_factor, false, 1)
  gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);

  #ifdef CODEGEN
  printf("gemmini_extended_config_ex(%d, %d, %d, %d, %s, %s);\n", dataflow, act & 3, 0, 1, a_transpose ? "true" : "false", b_transpose ? "true" : "false");
  printf("gemmini_extended_config_st(%d, %d, %f);\n", stride_C * sizeof_C, act & 3, scale);
  printf("gemmini_extended3_config_ld(%d, %f, %s, %d);\n", stride_A * sizeof(elem_t), A_scale_factor, "false", 0);
  printf("gemmini_extended3_config_ld(%d, %f, %s, %d);\n", stride_B * sizeof(elem_t), B_scale_factor, "false", 1);
  printf("gemmini_extended3_config_ld(%d, %f, %s, %d);\n", repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D ? "true" : "false", 2);
  #endif

  const size_t i0 = 0;
  const size_t j0 = 0;
  const size_t k0 = 0;
  const int a_spad_id = 1;
  const int b_spad_id = 1;
  const void * pre = NULL;
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t*)D + (bias_row * stride_D + j0 * tile_J * DIM)*sizeof_D;

        void * out = k0 == K0-1 ? (int8_t*)C + (i0*tile_I*DIM*stride_C + j0*tile_J*DIM)*sizeof_C : NULL;
        // print("out: %p, C: %p\n", out, C);

        const size_t I = last_I;
        const size_t J = last_J;
        const size_t K = last_K;

        const size_t pad_I =  padding_I;
        const size_t pad_J =  padding_J;
        const size_t pad_K =  padding_K;

        // printf("no_bias: %d, a_reuse: %d, b_reuse: %d, a_spad_id: %d, b_spad_id: %d, a: %llu, b: %llu \n",  no_bias, a_reuse, b_reuse, a_spad_id, b_spad_id, a, b);
        // printf("I: %d, J: %d, K: %d, pad_I: %d, pad_J: %d, pad_K: %d\n", I, J, K, pad_I, pad_J, pad_K);
        // if(pad_J != 3) {
        //   printf("!!!!! pad_J = %d\n", pad_J);
        // }
        // printf("I: %d, J: %d, K: %d, pad_I: %d, pad_J: %d, pad_K: %d\n", I, J, K, pad_I, pad_J, pad_K);
        // printf("i0: %d, j0: %d, k0: %d\n", i0, j0, k0);

        // printf("a: %p\tb: %p\tpre: %p\tout: %p\t"
        //    "I: %zu\tJ: %zu\tK: %zu\t"
        //    "pad_I: %zu\tpad_J: %zu\tpad_K: %zu\t"
        //    "stride_A: %zu\tstride_B: %zu\tstride_D: %zu\tstride_C: %zu\t"
        //    "a_transpose: %d\tb_transpose: %d\t"
        //    "full_C: %d\tlow_D: %d\t"
        //    "no_bias: %d\trepeating_bias: %d\t"
        //    "act: %d\ta_spad_id: %d\tb_spad_id: %d\n",
        //    (void *)a, (void *)b, (void *)pre, (void *)out,
        //    I, J, K,
        //    pad_I, pad_J, pad_K,
        //    stride_A, stride_B, stride_D, stride_C,
        //    a_transpose, b_transpose,
        //    full_C, low_D,
        //    no_bias, repeating_bias,
        //    act, a_spad_id, b_spad_id);

        dram_sp_tiled_matmul_ws(sp_A_addr, B, pre, out,
            A_scale_factor, B_scale_factor, D_scale_factor,
            I, J, K,
            pad_I, pad_J, pad_K,
            stride_A, stride_B, stride_D, stride_C,
            a_transpose, b_transpose,
            full_C, low_D,
            no_bias, repeating_bias,
            act, a_spad_id, b_spad_id);

  gemmini_fence();
}

static void tiled_matmul_outer_simple_dram_dram(size_t dim_I, size_t dim_J, size_t dim_K,
        const uint32_t sp_A_addr, const uint32_t sp_B_addr,
        const void * D, const uint32_t sp_C_addr,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA,
        int dataflow) {

  // printf("dim_I: %d\tdim_J: %d\tdim_K: %d\ttile_I: %d\ttile_J: %d\ttile_K: %d\n",
          //  dim_I, dim_J, dim_K, tile_I, tile_J, tile_K);
  const size_t dim_I_padded = tile_I * DIM;
  const size_t dim_J_padded = tile_J * DIM;
  const size_t dim_K_padded = tile_K * DIM;

  // const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
  // const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  // const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);

  const size_t I0 = 1;
  const size_t J0 = 1;
  const size_t K0 = 1;

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I*DIM) == 0 ? tile_I : (dim_I_padded/DIM) % tile_I;
  const size_t last_J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I;
  const size_t padding_J = dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias) {
    D = (void*) 1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose, b_transpose);
  gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
  // gemmini_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
  gemmini_extended3_config_ld(stride_B * sizeof(elem_t), B_scale_factor, false, 1)
  gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);

  #ifdef CODEGEN
  printf("gemmini_extended_config_ex(%d, %d, %d, %d, %s, %s);\n", dataflow, act & 3, 0, 1, a_transpose ? "true" : "false", b_transpose ? "true" : "false");
  printf("gemmini_extended_config_st(%d, %d, %f);\n", stride_C * sizeof_C, act & 3, scale);
  printf("gemmini_extended3_config_ld(%d, %f, %s, %d);\n", stride_A * sizeof(elem_t), A_scale_factor, "false", 0);
  printf("gemmini_extended3_config_ld(%d, %f, %s, %d);\n", stride_B * sizeof(elem_t), B_scale_factor, "false", 1);
  printf("gemmini_extended3_config_ld(%d, %f, %s, %d);\n", repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D ? "true" : "false", 2);
  #endif

  const size_t i0 = 0;
  const size_t j0 = 0;
  const size_t k0 = 0;
  const int a_spad_id = 1;
  const int b_spad_id = 1;
  const void * pre = NULL;
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t*)D + (bias_row * stride_D + j0 * tile_J * DIM)*sizeof_D;

        void * out = k0 == K0-1 ? (int8_t*)C + (i0*tile_I*DIM*stride_C + j0*tile_J*DIM)*sizeof_C : NULL;
        // print("out: %p, C: %p\n", out, C);

        const size_t I = last_I;
        const size_t J = last_J;
        const size_t K = last_K;

        const size_t pad_I =  padding_I;
        const size_t pad_J =  padding_J;
        const size_t pad_K =  padding_K;

        // printf("no_bias: %d, a_reuse: %d, b_reuse: %d, a_spad_id: %d, b_spad_id: %d, a: %llu, b: %llu \n",  no_bias, a_reuse, b_reuse, a_spad_id, b_spad_id, a, b);
        // printf("I: %d, J: %d, K: %d, pad_I: %d, pad_J: %d, pad_K: %d\n", I, J, K, pad_I, pad_J, pad_K);
        // if(pad_J != 3) {
        //   printf("!!!!! pad_J = %d\n", pad_J);
        // }
        // printf("I: %d, J: %d, K: %d, pad_I: %d, pad_J: %d, pad_K: %d\n", I, J, K, pad_I, pad_J, pad_K);
        // printf("i0: %d, j0: %d, k0: %d\n", i0, j0, k0);

        // printf("a: %p\tb: %p\tpre: %p\tout: %p\t"
        //    "I: %zu\tJ: %zu\tK: %zu\t"
        //    "pad_I: %zu\tpad_J: %zu\tpad_K: %zu\t"
        //    "stride_A: %zu\tstride_B: %zu\tstride_D: %zu\tstride_C: %zu\t"
        //    "a_transpose: %d\tb_transpose: %d\t"
        //    "full_C: %d\tlow_D: %d\t"
        //    "no_bias: %d\trepeating_bias: %d\t"
        //    "act: %d\ta_spad_id: %d\tb_spad_id: %d\n",
        //    (void *)a, (void *)b, (void *)pre, (void *)out,
        //    I, J, K,
        //    pad_I, pad_J, pad_K,
        //    stride_A, stride_B, stride_D, stride_C,
        //    a_transpose, b_transpose,
        //    full_C, low_D,
        //    no_bias, repeating_bias,
        //    act, a_spad_id, b_spad_id);

        spad_spad_tiled_matmul_ws(sp_A_addr, sp_B_addr, pre, sp_C_addr,
            A_scale_factor, B_scale_factor, D_scale_factor,
            I, J, K,
            pad_I, pad_J, pad_K,
            stride_A, stride_B, stride_D, stride_C,
            a_transpose, b_transpose,
            full_C, low_D,
            no_bias, repeating_bias,
            act, a_spad_id, b_spad_id);

  gemmini_fence();
}
