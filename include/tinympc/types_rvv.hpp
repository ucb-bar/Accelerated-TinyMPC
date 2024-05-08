#pragma once
#ifndef TINYMPC_TYPES_RVV_H
#define TINYMPC_TYPES_RVV_H

#include <cstdlib>
#include <cstdio>
#include <assert.h>

#include "glob_opts.hpp"
#include "matlib/matlib.h"

#ifdef RVV_DEFAULT_TO_ROW_MAJOR
#define RVV_DEFAULT_MATRIX_STORAGE_ORDER_OPTION RowMajor
#else
#define RVV_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ColMajor
#endif

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

    // Copy Constructor
    Matrix(const Matrix& other) {
        assert(other.rows <= MaxRows_ && other.cols <= MaxCols_);
        _Matrix(Rows_, Cols_);
        matcopy(data, other.data, outer, inner);
    }

    // Copy Constructor
    Matrix(Scalar_ *data) {
        _Matrix(Rows_, Cols_);
        matsetv(this->data, data, outer, inner);
    }

#if defined(USE_RVV) || defined(USE_CPU) || defined(USE_GEMMINI)
    // Column if ColMajor
    Scalar_ *col(int col) {
        assert(!(Options_ & RowMajor));
        return vector[col];
    }

    // Row if RowMajor
    Scalar_ *row(int row) {
        assert(Options_ & RowMajor);
        return vector[row];
    }
#endif
#ifdef USE_RVA
    // Column if ColMajor
    Scalar_ **col(int col) {
        assert(!(Options_ & RowMajor));
        return &vector[col];
    }

    // Row if RowMajor
    Scalar_ **row(int row) {
        assert(Options_ & RowMajor);
        return &vector[row];
    }
#endif

    // Assignment Operator
    // TODO: it has a bug in the last statement
    virtual Matrix& operator=(const Matrix *other) {
        if (this == other) return *this;
        matcopy(other->data, data, outer, inner);
        return *this;
    }

    // Assignment Operator
    virtual Matrix& operator=(const Scalar_ f) {
        matset(data, f, outer, inner);
        return *this;
    }

    // Assignment Operator
    Matrix& set(Scalar_ *f) {
        matsetv(data, f, outer, inner);
        return *this;
    }

    // Access Operator
    Scalar_& operator()(int row, int col) {
        // Access elements based on storage order
        if (Options_ & RowMajor) {
            return array[col][row];
        } else {
            return array[row][col];
        }
    }

    Scalar_ checksum() {
        Scalar_ sum = 0;
        for (int i = 0; i < outer; i++) {
            for (int j = 0; j < inner; j++) {
                sum += array[i][j];
	    }
        }
        return sum;
    }

    void print(const char *type, const char *name) {
        print_array_2d(data, outer, inner, type, name);
    }

    virtual void toString() {
        printf("const array: %x rows: %d cols: %d outer: %d inner: %d (%d, %d)\n", data, rows, cols, outer, inner, Rows_, Cols_);
    }
};

#ifdef __cplusplus
extern "C" {
#endif

typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;
typedef Matrix<tinytype, NINPUTS, 1> tiny_VectorNu;
typedef Matrix<tinytype, NSTATES, NSTATES, RowMajor> tiny_MatrixNxNx;
typedef Matrix<tinytype, NSTATES, NINPUTS, RowMajor> tiny_MatrixNxNu;
typedef Matrix<tinytype, NINPUTS, NSTATES, RowMajor> tiny_MatrixNuNx;
typedef Matrix<tinytype, NINPUTS, NINPUTS, RowMajor> tiny_MatrixNuNu;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;       // Nx x Nh
typedef Matrix<tinytype, NINPUTS, NHORIZON - 1> tiny_MatrixNuNhm1; // Nu x Nh-1

/**
 * Matrices that must be recomputed with changes in time step, rho
 */

#ifdef USE_GEMMINI
typedef uint32_t spad_ptr_t;
#define SPAD_ROWS(A) (((A) + (DIM) - 1) / (DIM))
#define SPAD_VEC_ROWS(A) (A)

const static spad_ptr_t cache_base_ptr = 0;
const static spad_ptr_t I_spad = cache_base_ptr;
const static spad_ptr_t nI_spad = I_spad + DIM;
const static spad_ptr_t rI_spad = nI_spad + DIM;
const static spad_ptr_t nrI_spad = rI_spad + DIM;
const static spad_ptr_t Kinf_spad = nrI_spad + DIM;
const static spad_ptr_t KinfT_spad = Kinf_spad + SPAD_ROWS(NINPUTS * NSTATES);
const static spad_ptr_t Pinf_spad = KinfT_spad + SPAD_ROWS(NINPUTS * NSTATES);
const static spad_ptr_t PinfT_spad = Pinf_spad + SPAD_ROWS(NSTATES * NSTATES);
const static spad_ptr_t Quu_inv_spad = PinfT_spad + SPAD_ROWS(NSTATES * NSTATES);
const static spad_ptr_t AmBKt_spad = Quu_inv_spad + SPAD_ROWS(NINPUTS * NINPUTS);
const static spad_ptr_t coeff_d2p_spad = AmBKt_spad + SPAD_ROWS(NSTATES * NSTATES);
const static spad_ptr_t Q_spad = coeff_d2p_spad + SPAD_ROWS(NSTATES * NINPUTS);
const static spad_ptr_t Q_mat_spad = coeff_d2p_spad + SPAD_ROWS(NSTATES * DIM);
const static spad_ptr_t Qf_spad = Q_mat_spad + SPAD_ROWS(NSTATES);
const static spad_ptr_t R_spad = Qf_spad + SPAD_ROWS(NSTATES);
const static spad_ptr_t Adyn_spad = R_spad + SPAD_ROWS(NINPUTS);
const static spad_ptr_t AdynT_spad = Adyn_spad + SPAD_ROWS(NSTATES * NSTATES);
const static spad_ptr_t Bdyn_spad = AdynT_spad + SPAD_ROWS(NSTATES * NSTATES);
const static spad_ptr_t BdynT_spad = Bdyn_spad + SPAD_ROWS(NSTATES * NINPUTS);

const static spad_ptr_t work_base_ptr = BANK_ROWS;
const static spad_ptr_t x_spad = work_base_ptr;
const static spad_ptr_t u_spad = x_spad + SPAD_VEC_ROWS(NSTATES * NHORIZON);
const static spad_ptr_t q_spad = u_spad + SPAD_VEC_ROWS(NINPUTS * (NHORIZON-1));
const static spad_ptr_t r_spad = q_spad + SPAD_VEC_ROWS(NSTATES * NHORIZON);
const static spad_ptr_t p_spad = r_spad + SPAD_VEC_ROWS(NINPUTS * (NHORIZON-1));
const static spad_ptr_t d_spad = p_spad + SPAD_VEC_ROWS(NSTATES * NHORIZON);
const static spad_ptr_t v_spad = d_spad + SPAD_VEC_ROWS(NINPUTS * (NHORIZON-1));
const static spad_ptr_t vnew_spad = v_spad + SPAD_VEC_ROWS(NSTATES * NHORIZON);
const static spad_ptr_t z_spad = vnew_spad + SPAD_VEC_ROWS(NSTATES * NHORIZON);
const static spad_ptr_t znew_spad = z_spad + SPAD_VEC_ROWS(NINPUTS * (NHORIZON-1));
const static spad_ptr_t g_spad = znew_spad + SPAD_VEC_ROWS(NINPUTS * (NHORIZON-1));
const static spad_ptr_t y_spad = g_spad + SPAD_VEC_ROWS(NSTATES * NHORIZON);
const static spad_ptr_t u_min_spad = y_spad + SPAD_VEC_ROWS(NINPUTS * (NHORIZON-1));
const static spad_ptr_t u_max_spad = u_min_spad + SPAD_VEC_ROWS(NINPUTS * (NHORIZON-1));
const static spad_ptr_t x_min_spad = u_max_spad + SPAD_VEC_ROWS(NINPUTS * (NHORIZON-1));
const static spad_ptr_t x_max_spad = x_min_spad + SPAD_VEC_ROWS(NSTATES * NHORIZON);
const static spad_ptr_t Xref_spad = x_max_spad + SPAD_VEC_ROWS(NSTATES * NHORIZON);
const static spad_ptr_t Uref_spad = Xref_spad + SPAD_VEC_ROWS(NSTATES * NHORIZON);

const static spad_ptr_t temp_spad = 2*BANK_ROWS;
#endif

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
