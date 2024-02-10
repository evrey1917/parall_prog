#include <iostream>
#include <cmath>

#ifdef USE_DOUBLE
#define NAME double
#else
#define NAME float
#endif

int main()
{
    NAME pi = acos(-1), sum_of_sinus = 0, n = 10000000, delta = 1/n, sin_arg = 0;

    for (int i = 0; i < n; i++)
    {
        sum_of_sinus = sum_of_sinus + sin(2*pi*sin_arg);
        sin_arg = sin_arg + delta;
    }

    std::cout << typeid(sum_of_sinus).name() << " sum of sinus in [0; 2pi]: " << sum_of_sinus << std::endl;

    return 0;
}