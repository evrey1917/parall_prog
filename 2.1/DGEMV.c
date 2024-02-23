#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

/*
 * matrix_vector_product: Compute matrix-vector product c[m] = a[m][n] * b[n]
 */
void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

// void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n)
// {
// #pragma omp parallel
//     {
//         double t = omp_get_wtime();
//         int nthreads = omp_get_num_threads();
//         int threadid = omp_get_thread_num();
//         int items_per_thread = m / nthreads;
//         int lb = threadid * items_per_thread;
//         int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
//         for (int i = lb; i <= ub; i++)
//         {
//             c[i] = 0.0;
//             for (int j = 0; j < n; j++)
//                 c[i] += a[i * n + j] * b[j];
//         }
//         t = omp_get_wtime() - t;
//         printf("Thread %d items %d [%d - %d], time: %.6f\n", threadid, ub - lb + 1, lb, ub, t);
//     }
// }

/*
    matrix_vector_product_omp: Compute matrix-vector product c[m] = a[m][n] * b[n]
*/
void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n)
{
#pragma omp parallel
    {
        // from lectors guide

        // int nthreads = omp_get_num_threads();
        // int threadid = omp_get_thread_num();
        // int items_per_thread = m / nthreads;
        // int lb = threadid * items_per_thread;
        // int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        // for (int i = lb; i <= ub; i++)

# pragma omp for schedule(static)   // from documentation
        for (int i = 0; i < m; i++)
        {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

double run_serial(size_t n, size_t m)
{
    double *a, *b, *c;
    a = (double*)malloc(sizeof(*a) * m * n);
    b = (double*)malloc(sizeof(*b) * n);
    c = (double*)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
    {
        free(a);
        free(b);
        free(c);
        printf("Error allocate memory!\n");
        exit(1);
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (int j = 0; j < n; j++)
        b[j] = j;

    double t = cpuSecond();
    matrix_vector_product(a, b, c, m, n);
    t = cpuSecond() - t;

    printf("Elapsed time (serial): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);

    return t;
}

double run_parallel(size_t n, size_t m)
{
    double *a, *b, *c;

    a = (double*)malloc(sizeof(*a) * m * n);
    b = (double*)malloc(sizeof(*b) * n);
    c = (double*)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
    {
        free(a);
        free(b);
        free(c);
        printf("Error allocate memory!\n");
        exit(1);
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (int j = 0; j < n; j++)
    {
        b[j] = j;
    }

    double t = cpuSecond();
    matrix_vector_product_omp(a, b, c, m, n);
    t = cpuSecond() - t;

    printf("Elapsed time (parallel): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);

    return t;
}

int main(int argc, char *argv[])
{
    size_t N = 20000;
    if (argc > 1)
    {
        N = atoi(argv[1]);
    }

    double time_serial = run_serial(N, N);

    int mas[8] = {1,2,4,7,8,16,20,40};

    for (int i = 0; i < 8; i++)
    {
        omp_set_num_threads(mas[i]);
        printf("On %d threads: %.6f\n", mas[i], time_serial/run_parallel(N, N));
    }
    return 0;
}
