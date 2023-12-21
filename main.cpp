#include "include/complex.hpp"
#include "include/real.hpp"

#include "include/tensor.hpp"

#include <iostream>

template <typename T>
void printVector(const std::vector<T> &vec)
{
    for (const T &value : vec)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

int main()
{
    Tensor<Complex<float>> T = Tensor<Complex<float>>({2, 5});
    printVector(T.shape());

    std::cout << T << std::endl;
    std::cout << T[{0, 0}] << std::endl;
    T[{0, 0}] = Complex<float>(2, 2);
    std::cout << T * Complex<float>(1, 3) << std::endl;

    return 0;
}
