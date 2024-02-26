#pragma once
// common.h
// common utilities for the test code under examples/
#ifndef RVV_MATLIB_COMMON_H
#define RVV_MATLIB_COMMON_H

extern "C" {

void gen_rand_1d(float *a, int n);
void gen_string(char *s, int n);
void gen_rand_2d(float **ar, int n, int m);
void print_string(const char *a, const char *name);
void print_array_1d(float *a, int n, const char *type, const char *name);
void print_array_2d(float **a, int n, int m, const char *type, const char *name);
bool float_eq(float golden, float actual, float relErr);
bool compare_1d(float *golden, float *actual, int n);
bool compare_string(const char *golden, const char *actual, int n);
bool compare_2d(float **golden, float **actual, int n, int m);
float **alloc_array_2d(int n, int m);
float **alloc_array_2d_col(int n, int m);
float *alloc_array_1d(int n);
void free_array_2d(float **ar);
void free_array_1d(float *ar);
void init_array_zero_1d(float *ar, int n);
void init_array_one_1d(float *ar, int n);
void init_array_one_2d(float **ar, int n, int m);

void printx(float **a, int n, int m, const char *name);
};
#endif