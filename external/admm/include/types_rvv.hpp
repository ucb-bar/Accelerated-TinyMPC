#pragma once
#ifndef TINYMPC_TYPES_RVV_H
#define TINYMPC_TYPES_RVV_H

// #include <cstdlib>

#include <assert.h>

#include <matlib.h>

#ifndef NSTATES
#define NSTATES 12
#endif
#ifndef NINPUTS
#define NINPUTS 4
#endif
#ifndef NHORIZON
#define NHORIZON 10
#endif
#ifndef NTOTAL
#define NTOTAL 301
#endif


#define RVV_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ColMajor


typedef float tinytype;


enum StorageOptions {
    /** Storage order is column major (see \ref TopicStorageOrders). */
    ColMajor = 0,
    /** Storage order is row major (see \ref TopicStorageOrders). */
    RowMajor = 0x1,  // it is only a coincidence that this is equal to RowMajorBit -- don't rely on that
    /** Align the matrix itself if it is vectorizable fixed-size */
    AutoAlign = 0,
    DontAlign = 0x2
};

// Forward declarations
template<typename Scalar_, int Rows_, int Cols_,
        int Options_ = AutoAlign |
                       ( (Cols_ == 1 && Rows_ > 1) ? ColMajor
                       : (Rows_ == 1 && Cols_ > 1) ? RowMajor
                       : RVV_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
        int MaxRows_ = Rows_,
        int MaxCols_ = Cols_
> class Matrix {

public:
    Scalar_ _data[MaxRows_ * MaxCols_];
    Scalar_ *data;
    Scalar_ *vector[Options_ & RowMajor ? Rows_ : Cols_];
    Scalar_ **array;
    int rows, cols, outer, inner;

    void _Matrix(int rows_, int cols_) {
        rows = rows_;
        cols = cols_;
        if (Options_ & RowMajor) {
            outer = rows_;
            inner = cols_;
        } else {
            outer = cols_;
            inner = rows_;
        }
        data = _data;
        array = &vector[0];
        for (int i = 0; i < outer; ++i)
            array[i] = (Scalar_ *)(&_data[i * inner]);
    }

    // Constructor
    Matrix() {
        _Matrix(Rows_, Cols_);
        for (int i = 0; i < outer * inner; ++i)
            data[i] = 0;
    }







};

#ifdef __cplusplus
extern "C" {
#endif




// ======== Matrix Structs =========

// Vector of NSTATES x 1 (ColMajor)
typedef struct {
    tinytype data[NSTATES];
    tinytype* vector[1];  // only 1 column
    tinytype** array;
    int rows, cols;
    int outer, inner;
} tiny_VectorNx;

// Vector of NINPUTS x 1 (ColMajor)
typedef struct {
    tinytype data[NINPUTS];
    tinytype* vector[1];
    tinytype** array;
    int rows, cols;
    int outer, inner;
} tiny_VectorNu;

// Matrix NSTATES x NSTATES (RowMajor)
typedef struct {
    tinytype data[NSTATES * NSTATES];
    tinytype* vector[NSTATES];  // one row per entry
    tinytype** array;
    int rows, cols;
    int outer, inner;
} tiny_MatrixNxNx;

// Matrix NSTATES x NINPUTS (RowMajor)
typedef struct {
    tinytype data[NSTATES * NINPUTS];
    tinytype* vector[NSTATES];
    tinytype** array;
    int rows, cols;
    int outer, inner;
} tiny_MatrixNxNu;

// Matrix NINPUTS x NSTATES (RowMajor)
typedef struct {
    tinytype data[NINPUTS * NSTATES];
    tinytype* vector[NINPUTS];
    tinytype** array;
    int rows, cols;
    int outer, inner;
} tiny_MatrixNuNx;

// Matrix NINPUTS x NINPUTS (RowMajor)
typedef struct {
    tinytype data[NINPUTS * NINPUTS];
    tinytype* vector[NINPUTS];
    tinytype** array;
    int rows, cols;
    int outer, inner;
} tiny_MatrixNuNu;

// Matrix NSTATES x NHORIZON (ColMajor)
typedef struct {
    tinytype data[NSTATES * NHORIZON];
    tinytype* vector[NHORIZON];  // each column
    tinytype** array;
    int rows, cols;
    int outer, inner;
} tiny_MatrixNxNh;

// Matrix NINPUTS x (NHORIZON - 1) (ColMajor)
typedef struct {
    tinytype data[NINPUTS * (NHORIZON - 1)];
    tinytype* vector[NHORIZON - 1];
    tinytype** array;
    int rows, cols;
    int outer, inner;
} tiny_MatrixNuNhm1;

// ======== Init Helpers =========

static inline void init_col_major_layout(tinytype* data, tinytype** vector, int outer, int inner) {
    for (int i = 0; i < outer; ++i)
        vector[i] = &data[i * inner];
}

static inline void init_row_major_layout(tinytype* data, tinytype** vector, int outer, int inner) {
    for (int i = 0; i < outer; ++i)
        vector[i] = &data[i * inner];
}
static inline void init_VectorNx(tiny_VectorNx* mat) {
    mat->rows = NSTATES;
    mat->cols = 1;
    mat->outer = 1;
    mat->inner = NSTATES;
    mat->array = mat->vector;
    init_col_major_layout(mat->data, mat->vector, mat->outer, mat->inner);
}

static inline void init_VectorNu(tiny_VectorNu* mat) {
    mat->rows = NINPUTS;
    mat->cols = 1;
    mat->outer = 1;
    mat->inner = NINPUTS;
    mat->array = mat->vector;
    init_col_major_layout(mat->data, mat->vector, mat->outer, mat->inner);
}

static inline void init_MatrixNxNx(tiny_MatrixNxNx* mat) {
    mat->rows = NSTATES;
    mat->cols = NSTATES;
    mat->outer = NSTATES;
    mat->inner = NSTATES;
    mat->array = mat->vector;
    init_row_major_layout(mat->data, mat->vector, mat->outer, mat->inner);
}

static inline void init_MatrixNxNu(tiny_MatrixNxNu* mat) {
    mat->rows = NSTATES;
    mat->cols = NINPUTS;
    mat->outer = NSTATES;
    mat->inner = NINPUTS;
    mat->array = mat->vector;
    init_row_major_layout(mat->data, mat->vector, mat->outer, mat->inner);
}

static inline void init_MatrixNuNx(tiny_MatrixNuNx* mat) {
    mat->rows = NINPUTS;
    mat->cols = NSTATES;
    mat->outer = NINPUTS;
    mat->inner = NSTATES;
    mat->array = mat->vector;
    init_row_major_layout(mat->data, mat->vector, mat->outer, mat->inner);
}

static inline void init_MatrixNuNu(tiny_MatrixNuNu* mat) {
    mat->rows = NINPUTS;
    mat->cols = NINPUTS;
    mat->outer = NINPUTS;
    mat->inner = NINPUTS;
    mat->array = mat->vector;
    init_row_major_layout(mat->data, mat->vector, mat->outer, mat->inner);
}

static inline void init_MatrixNxNh(tiny_MatrixNxNh* mat) {
    mat->rows = NSTATES;
    mat->cols = NHORIZON;
    mat->outer = NHORIZON;
    mat->inner = NSTATES;
    mat->array = mat->vector;
    init_col_major_layout(mat->data, mat->vector, mat->outer, mat->inner);
}

static inline void init_MatrixNuNhm1(tiny_MatrixNuNhm1* mat) {
    mat->rows = NINPUTS;
    mat->cols = NHORIZON - 1;
    mat->outer = NHORIZON - 1;
    mat->inner = NINPUTS;
    mat->array = mat->vector;
    init_col_major_layout(mat->data, mat->vector, mat->outer, mat->inner);
}



/**
 * Matrices that must be recomputed with changes in time step, rho
 */
typedef struct
{
    tinytype rho;
    tiny_MatrixNuNx Kinf;
    tiny_MatrixNxNu KinfT;
    tinytype * Kinf_data;
    tiny_MatrixNxNx Pinf;
    tiny_MatrixNxNx PinfT;
    tinytype * Pinf_data;
    tiny_MatrixNuNu Quu_inv;
    tinytype * Quu_inv_data;
    tiny_MatrixNxNx AmBKt;
    tinytype * AmBKt_data;
    tiny_MatrixNxNu coeff_d2p;
} TinyCache;

/**
 * User settings
 */
typedef struct
{
    tinytype abs_pri_tol;
    tinytype abs_dua_tol;
    int max_iter;
    int check_termination;
    int en_state_bound;
    int en_input_bound;
} TinySettings;

/**
 * Problem variables
 */
typedef struct
{
    // State and input
    tiny_MatrixNxNh x;
    tiny_MatrixNuNhm1 u;

    // Linear control cost terms
    tiny_MatrixNxNh q;
    tiny_MatrixNuNhm1 r;

    // Linear Riccati backward pass terms
    tiny_MatrixNxNh p;
    tiny_MatrixNuNhm1 d;

    // Auxiliary variables
    tiny_MatrixNxNh v;
    tiny_MatrixNxNh vnew;
    tiny_MatrixNuNhm1 z;
    tiny_MatrixNuNhm1 znew;

    // Dual variables
    tiny_MatrixNxNh g;
    tiny_MatrixNuNhm1 y;

    tinytype primal_residual_state;
    tinytype primal_residual_input;
    tinytype dual_residual_state;
    tinytype dual_residual_input;
    int status;
    int iter;

    tiny_VectorNx Q;
    tiny_VectorNx Qf;
    tiny_VectorNu R;
    tiny_MatrixNxNx Adyn;
    tiny_MatrixNxNx AdynT;
    tinytype * Adyn_data;
    tiny_MatrixNxNu Bdyn;
    tiny_MatrixNuNx BdynT;
    tinytype * Bdyn_data;

    tiny_MatrixNuNhm1 u_min;
    tiny_MatrixNuNhm1 u_max;
    tiny_MatrixNxNh x_min;
    tiny_MatrixNxNh x_max;
    tiny_MatrixNxNh Xref;   // Nx x Nh
    tiny_MatrixNuNhm1 Uref; // Nu x Nh-1

    // Temporaries
    tiny_VectorNu Qu;
    tiny_VectorNu u1, u2;
    tiny_VectorNx x1, x2, x3;
    tiny_MatrixNuNhm1 m1, m2;
    tiny_MatrixNxNh s1, s2;
} TinyWorkspace;

/**
 * Main TinyMPC solver structure that holds all information.
 */
typedef struct
{
    TinySettings *settings; // Problem settings
    TinyCache *cache;       // Problem cache
    TinyWorkspace *work;    // Solver workspace
} TinySolver;

#ifdef __cplusplus
}
#endif
#endif
