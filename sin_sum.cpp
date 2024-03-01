#include <iostream>
#include <cmath>
#include <vector>

#ifdef USE_DOUBLE
#define NAME double
#else
#define NAME float
#endif

int main()
{
    NAME pi = acos(-1), sum_of_sinus = 0, sum_of_sinus_array = 0, n = 10000000, delta = 1/n, sin_arg = 0, sin_out;

    std::vector<NAME> vector_NAME;

    for (int i = 0; i < n; i++)
    {
        sin_out = sin(2*pi*sin_arg);
        vector_NAME.push_back(sin_out);

        sum_of_sinus_array = sum_of_sinus_array + vector_NAME[i];
        sum_of_sinus = sum_of_sinus + sin_out;

        sin_arg = delta * i;
    }

    std::cout << typeid(sum_of_sinus).name() << " sum of sinus in [0; 2pi]: " << sum_of_sinus << std::endl;
    std::cout << typeid(sum_of_sinus).name() << " sum of sinus in [0; 2pi] from array: " << sum_of_sinus << std::endl;

    return 0;
}
