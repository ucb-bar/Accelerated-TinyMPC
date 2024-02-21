#pragma once
#ifndef TINYMPC_TYPES_RVV_H
#define TINYMPC_TYPES_RVV_H

#include <cstdlib>
#include <stdio.h>
#include "common.h"
#include "rvv_matlib.h"
#include "glob_opts.hpp"

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
    // DontAlign = 0x2
};

// Forward declarations
template<typename Scalar_, int Rows_, int Cols_,
        int Options_ = AutoAlign |
                       ( (Cols_ == 1 && Rows_ > 1) ? ColMajor
                       : (Rows_ == 1 && Cols_ > 1) ? RowMajor
                       : RVV_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
        int MaxRows_ = Rows_,
        int MaxCols_ = Cols_
> class Matrix;

template<typename Scalar_, int Rows_> class MatrixCol;

// Actual definitions
template<typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_> class Matrix {

public:
    Scalar_ **data;
    int rows, cols, outer, inner;
    int reference = 1;
    bool column = false;

    void init() {
        if (Options_ & RowMajor) {
            outer = Rows_;
            inner = Cols_;
        } else {
            inner = Rows_;
            outer = Cols_;
        }
    }

    // Constructor
    Matrix() {
        rows = Rows_;
        cols = Cols_;
        init();
        if (MaxCols_ > 0 && MaxRows_ > 0) {
            data = alloc_array_2d(outer, inner);
        }
        // toString();
    }

    // Copy Constructor
    Matrix(const Matrix& other) {
        matcopy(data, other.data, outer, inner);
        rows = other.rows;
        cols = other.cols;
        init();
    }

    // Copy Constructor
    Matrix(Scalar_ *data) {
        matsetv(this->data, data, outer, inner);
        rows = Rows_;
        cols = Cols_;
        init();
    }

    // Destructor
    ~Matrix() {
        if (--reference == 0) free_array_2d(data);
    }

    // Column
    Matrix<Scalar_, Rows_, 1, ColMajor, 0, 0>& col(int col) {
        Matrix<Scalar_, Rows_, 1, ColMajor, 0, 0> *_col = new Matrix<Scalar_, Rows_, 1, ColMajor, 0, 0>();
        Scalar_ **target = { &data[col] };
        _col->data = target;
        _col->reference = _col->reference + 1;
        _col->column = true;
        return *_col;
    }

    // Assignment Operator
    virtual Matrix& operator=(const Matrix *other) {
        if (this == other) return *this;
        matcopy(data, other->data, outer, inner);
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
        // print_array_2d(data, outer, inner, "float", "data");
        return *this;
    }

    // Access Operator
    Scalar_& operator()(int row, int col) {
        // Access elements based on storage order
        if (Options_ & RowMajor) {
            return data[col][row];
        } else {
            return data[row][col];
        }
    }

    virtual void toString() {
        printf("const data: %x rows: %d cols: %d inner: %d outer: %d ref: %d (%d, %d)\n", data, rows, cols, inner, outer, reference, Rows_, Cols_);
    }
};

template<typename Scalar_, int Rows_>
class MatrixCol : public Matrix<Scalar_, Rows_, 1, ColMajor, 0, 0> {

public:
    MatrixCol<Scalar_, Rows_>& operator=(const Matrix<Scalar_, Rows_, 1> other) {
        matcopy(this->data, other.data, 1, Rows_);
        return *this;
    }

    // Assignment Operator
    MatrixCol<Scalar_, Rows_>& operator=(const Scalar_ f) {
        matset(this->data, f, 1, Rows_);
        return *this;
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
    typedef struct
    {
        tinytype rho;
        tiny_MatrixNuNx Kinf;
        tinytype * Kinf_data;
        tiny_MatrixNxNx Pinf;
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
        tinytype * Adyn_data;
        tiny_MatrixNxNu Bdyn;
        tinytype * Bdyn_data;

        tiny_MatrixNuNhm1 u_min;
        tiny_MatrixNuNhm1 u_max;
        tiny_MatrixNxNh x_min;
        tiny_MatrixNxNh x_max;
        tiny_MatrixNxNh Xref;   // Nx x Nh
        tiny_MatrixNuNhm1 Uref; // Nu x Nh-1

        // Temporaries
        tiny_VectorNu Qu;
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
