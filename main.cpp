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
    Tensor<Complex<float>> T = Tensor<Complex<float>>({2, 2, 2});
    printVector(T.shape());

    return 0;
}
