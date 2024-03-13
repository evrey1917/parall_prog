#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#define SCH_PARAM static
// #define SCH_PARAM dynamic
// #define SCH_PARAM guided

int number = 1;

const double epsilon = 0.00001;
const double tau = 0.0005;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void matrix_vector_product(double *a, double *b, double *matrix_prod_x, int n)
{
    for (int i = 0; i < n; i++)
    {
        matrix_prod_x[i] = 0;
        for (int j = 0; j < n; j++)
        {
            matrix_prod_x[i] += a[i * n + j] * b[j];
        }
    }
}

int quality_check(double *matrix_prod_x, double *b, double *x, int n)
{
    double summa_up = 0, summa_down = 0;

    for (int i = 0; i < n; i++)
    {
        summa_up += (matrix_prod_x[i] - b[i]) * (matrix_prod_x[i] - b[i]);
        summa_down += b[i] * b[i];
    }

    if (sqrt(summa_up)/sqrt(summa_down) - epsilon < 0)
    {
        return 1;
    }
    return 0;
}

void simple_iteration_step(double *a, double *b, double *x, double *matrix_prod_x, int n)
{
    matrix_vector_product(a, x, matrix_prod_x, n);

    if (quality_check(matrix_prod_x, b, x, n))
    {
        return;
    }

    for (int i = 0; i < n; i++)
    {
        x[i] = x[i] - tau * (matrix_prod_x[i] - b[i]);
    }

    // printf("%.6f\n", x[0]);

    simple_iteration_step(a, b, x, matrix_prod_x, n);
}

double run_serial(size_t n)
{
    double *a, *b, *x, *matrix_prod_x;
    a = (double*)malloc(sizeof(*a) * n * n);
    b = (double*)malloc(sizeof(*b) * n);
    x = (double*)malloc(sizeof(*x) * n);
    matrix_prod_x = (double*)malloc(sizeof(*matrix_prod_x) * n);

    if (a == NULL || b == NULL || x == NULL || matrix_prod_x == NULL)
    {
        free(a);
        free(b);
        free(x);
        free(matrix_prod_x);
        printf("Error allocate memory!\n");
        exit(1);
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                a[i * n + j] = 2;
            }
            else
            {
                a[i * n + j] = 1;
            }
        }
    }

    for (int j = 0; j < n; j++)
    {
        b[j] = n + 1;
        x[j] = 0;
    }

    double t = cpuSecond();
    simple_iteration_step(a, b, x, matrix_prod_x, n);
    t = cpuSecond() - t;

    printf("Elapsed time (serial): %.6f sec.\n\n", t);

    free(a);
    free(b);
    free(x);
    free(matrix_prod_x);

    return t;
}

void simple_iteration_step_omp_secundus(double *a, double *b, double *x, double *matrix_prod_x, int n)
{
    double sub_summa_up = 0, sub_summa_down = 0;

    #pragma omp parallel
    {
    while (1)
    {
    # pragma omp for schedule(SCH_PARAM, number)
        for (int i = 0; i < n; i++)
        {
            matrix_prod_x[i] = 0;
            for (int j = 0; j < n; j++)
            {
                matrix_prod_x[i] += a[i * n + j] * x[j];
            }
        }

    # pragma omp for schedule(SCH_PARAM, number) reduction(+: sub_summa_up, sub_summa_down)
        for (int i = 0; i < n; i++)
        {
            sub_summa_up += (matrix_prod_x[i] - b[i]) * (matrix_prod_x[i] - b[i]);
            sub_summa_down += b[i] * b[i];
        }

        if (sqrt(sub_summa_up)/sqrt(sub_summa_down) - epsilon < 0)
        {
            break;
        }
        #pragma omp single
        {
            sub_summa_up = 0;
            sub_summa_down = 0;
        }

    # pragma omp for schedule(SCH_PARAM, number)
        for (int i = 0; i < n; i++)
        {
            x[i] = x[i] - tau * (matrix_prod_x[i] - b[i]);
        }
    }
    }
}

double run_parallel_secundus(size_t n)
{
    double *a, *b, *x, *matrix_prod_x;
    a = (double*)malloc(sizeof(*a) * n * n);
    b = (double*)malloc(sizeof(*b) * n);
    x = (double*)malloc(sizeof(*x) * n);
    matrix_prod_x = (double*)malloc(sizeof(*matrix_prod_x) * n);

    if (a == NULL || b == NULL || x == NULL || matrix_prod_x == NULL)
    {
        free(a);
        free(b);
        free(x);
        free(matrix_prod_x);
        printf("Error allocate memory!\n");
        exit(1);
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                a[i * n + j] = 2;
            }
            else
            {
                a[i * n + j] = 1;
            }
        }
    }

    for (int j = 0; j < n; j++)
    {
        b[j] = n + 1;
        x[j] = 0;
    }

    int mas[11] = {1,2,4,7,8,16,20,40,80,160,320};

    for (int i = 0; i < 11; i++)
    {
        // int i = 6;
        printf("On %d threads: ", mas[i]);
        omp_set_num_threads(mas[i]);

        double t = cpuSecond();
        simple_iteration_step_omp_secundus(a, b, x, matrix_prod_x, n);
        t = cpuSecond() - t;

        printf("Elapsed time (parallel): %.6f sec.\n", t);
        for (int j = 0; j < n; j++)
        {
            x[j] = 0;
        }
    }

    free(a);
    free(b);
    free(x);
    free(matrix_prod_x);

    return 0;
}

int main(int argc, char **argv)
{
    size_t N = 5;

    if (argc > 1)
    {
        N = atoi(argv[1]);
    }

    run_parallel_secundus(N);

    return 0;
}
