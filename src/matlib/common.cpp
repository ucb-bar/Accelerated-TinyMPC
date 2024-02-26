#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

void gen_rand_1d(float *a, int n) {
    for (int i = 0; i < n; ++i)
        a[i] = (float)rand() / (float)RAND_MAX + (float)(rand() % 1000);
}

void gen_string(char *s, int n) {
    // char value range: -128 ~ 127
    for (int i = 0; i < n - 1; ++i)
        s[i] = (char)(rand() % 127) + 1;
    s[n - 1] = '\0';
}

void gen_rand_2d(float **ar, int n, int m) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            ar[i][j] = (float)rand() / (float)RAND_MAX + (float)(rand() % 1000);
}

void print_string(const char *a, const char *name) {
    printf("const char *%s = \"", name);
    int i = 0;
    while (a[i] != 0)
        putchar(a[i++]);
    printf("\"\n");
    puts("");
}

void print_array_1d(float *a, int n, const char *type, const char *name) {
    printf("%s %s[%d] = {\n", type, name, n);
    for (int i = 0; i < n; ++i) {
        printf("% 8.4f%s", a[i], i != n - 1 ? "," : "};\n");
        if (i % 10 == 9)
            puts("");
    }
    puts("");
}

void print_array_2d(float **a, int n, int m, const char *type,
                    const char *name) {
    printf("%s %s[%d][%d] = {\n", type, name, n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            printf("% 8.4f", a[i][j]);
            if (j == m - 1)
                puts(i == n - 1 ? "};" : ",");
            else
                putchar(',');
        }
    }
    puts("");
}

bool float_eq(float golden, float actual, float relErr) {
    return (fabs((actual - golden) / actual) < relErr);
}

bool compare_1d(float *golden, float *actual, int n) {
    for (int i = 0; i < n; ++i)
        if (!float_eq(golden[i], actual[i], 1e-6))
            return false;
    return true;
}

bool compare_string(const char *golden, const char *actual, int n) {
    for (int i = 0; i < n; ++i)
        if (golden[i] != actual[i])
            return false;
    return true;
}

bool compare_2d(float **golden, float **actual, int n, int m) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            if (!float_eq(golden[i][j], actual[i][j], 1e-6))
                return false;
    return true;
}

// Row major allocation
float **alloc_array_2d(int n, int m) {
    float **ret = (float **)malloc(sizeof(float *) * n);
    float *data = (float *)malloc(sizeof(float) * n * m);
    for (int i = 0; i < n; ++i)
        ret[i] = (float *)(&data[i * m]);
    for (int i = 0; i < m * n; i++) {
        data[i] = 0;
    }
    return ret;
}

// Column major allocation
float **alloc_array_2d_col(int n, int m) {
    float **ret = (float **)malloc(sizeof(float *) * m);
    float *data = (float *)malloc(sizeof(float) * n * m);
    for (int i = 0; i < m; ++i)
        ret[i] = (float *)(&data[i * n]);
    for (int i = 0; i < m * n; i++) {
        data[i] = 0;
    }
    return ret;
}

void free_array_2d(float **ar) {
    free(ar[0]);
    free((float *)ar);
}

float *alloc_array_1d(int n) {
    float *ret = (float *)malloc(sizeof(float) * n);
    return ret;
}

void free_array_1d(float *ar) {
    free(ar);
}

void init_array_zero_1d(float *ar, int n) {
    for (int i = 0; i < n; ++i)
        ar[i] = 0;
}

void init_array_one_1d(float *ar, int n) {
    for (int i = 0; i < n; ++i)
        ar[i] = 1;
}

void init_array_one_2d(float **ar, int n, int m) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            ar[i][j] = 1;
}

void printx(float **a, int n, int m, const char *name) {
    printf("%s ", name);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            printf("% 8.4f", a[i][j]);
            if (j == m - 1)
                puts(i == n - 1 ? "" : ",");
            else
                putchar(',');
        }
    }
}

}
