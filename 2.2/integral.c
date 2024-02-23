#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double func(double x)
{
    return exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
    {
        sum += func(a + h * (i + 0.5));
    }

    sum *= h;

    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0, sub_sum = 0.0;

#pragma omp parallel
    {
        // from lectors guide

        // int nthreads = omp_get_num_threads();
        // int threadid = omp_get_thread_num();
        // int items_per_thread = n / nthreads;
        // int lb = threadid * items_per_thread;
        // int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

# pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
        {
            sub_sum += func(a + h * (i + 0.5));
        }

        #pragma omp atomic
            sum += sub_sum;
    }
    sum *= h;

    return sum;
}

double run_serial()
{
    double t = cpuSecond();
    double res = integrate(func, a, b, nsteps);
    t = cpuSecond() - t;
    printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

double run_parallel()
{
    double t = cpuSecond();
    double res = integrate_omp(func, a, b, nsteps);
    t = cpuSecond() - t;
    printf("Result (parallel): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

int main(int argc, char **argv)
{
    printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);
    double tserial = run_serial();
    double tparallel;

    int mas[8] = {1,2,4,7,8,16,20,40};

    printf("Execution time (serial): %.6f\n\n", tserial);

    for (int i = 0; i < 8; i++)
    {
        omp_set_num_threads(mas[i]);
        tparallel = run_parallel();
        printf("Execution time (parallel %d threads): %.6f\n", mas[i], tparallel);
        printf("Speedup: %.2f\n\n", tserial / tparallel);
    }
    return 0;
}
