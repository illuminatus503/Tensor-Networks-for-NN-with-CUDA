#include "include/complex.hpp"
#include "include/complex_linalg.hpp"

#include "include/tensor.hpp"

#include <iostream>

int main()
{
    Complex<int> c1_int(3, 2);
    Complex<int> c2_int(1, 7);

    std::cout << "Integer Complex c1 + c2 = " << c1_int + c2_int << std::endl;

    Complex<float> c1_float(3.5, 2.3);
    Complex<float> c2_float(1.2, 7.8);

    std::cout << "Float Complex c1 + c2 = " << c1_float + c2_float << std::endl;

    Complex<float> c1(3.0, 2.0);
    Complex<float> c2(1.0, 7.0);

    // Basic Arithmetic Operations
    std::cout << "c1 + c2 = " << c1 + c2 << std::endl;
    std::cout << "c1 - c2 = " << c1 - c2 << std::endl;
    std::cout << "c1 * c2 = " << c1 * c2 << std::endl;
    std::cout << "c1 / c2 = " << c1 / c2 << std::endl;

    // Magnitude of a Complex Number
    std::cout << "Magnitude of c1 = " << magnitude(c1) << std::endl;
    std::cout << "Magnitude of c2 = " << magnitude(c2) << std::endl;

    // Conjugate of a Complex Number
    std::cout << "Conjugate of c1 = " << conjugate(c1) << std::endl;
    std::cout << "Conjugate of c2 = " << conjugate(c2) << std::endl;

    // Defining a tensor
    Tensor<Complex<float>> tensor(2, 3);
    tensor.setElement(0, 0, Complex<float>(1.0, 2.0));
    tensor.setElement(0, 1, Complex<float>(3.0, 4.0));

    // Printing the tensor
    std::cout << "Tensor Contents:" << std::endl;
    std::cout << tensor; // Using the overloaded << operator

    tensor.setElement(0, 0, Complex<float>(1.0, 2.0));
    tensor.setElement(0, 1, Complex<float>(3.0, 4.0));
    tensor.setElement(1, 0, Complex<float>(5.0, 6.0));
    tensor.setElement(1, 1, Complex<float>(7.0, 8.0));

    Tensor<Complex<float>> negatedTensor = -tensor;

    std::cout << "Original Tensor:" << std::endl
              << tensor;
    std::cout << "Negated Tensor:" << std::endl
              << negatedTensor;

    Tensor<Complex<float>> tensor1(2, 2);
    tensor1.setElement(0, 0, Complex<float>(1.0, 2.0));
    tensor1.setElement(0, 1, Complex<float>(3.0, 4.0));
    tensor1.setElement(1, 0, Complex<float>(5.0, 6.0));
    tensor1.setElement(1, 1, Complex<float>(7.0, 8.0));

    Tensor<Complex<float>> tensor2(2, 2);
    tensor2.setElement(0, 0, Complex<float>(1.0, 1.0));
    tensor2.setElement(0, 1, Complex<float>(1.0, 1.0));
    tensor2.setElement(1, 0, Complex<float>(1.0, 1.0));
    tensor2.setElement(1, 1, Complex<float>(1.0, 1.0));

    Tensor<Complex<float>> sum = tensor1 + tensor2;
    Tensor<Complex<float>> difference = tensor1 - tensor2;

    std::cout << "Sum of Tensors:" << std::endl
              << sum;
    std::cout << "Difference of Tensors:" << std::endl
              << difference;

    return 0;
}
