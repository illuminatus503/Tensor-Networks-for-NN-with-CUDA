#include "include/complex.hpp"
#include "include/real.hpp"
#include "include/matrix.hpp"

#include <iostream>

int main()
{
    // ! Real number algebra
    // Real number algebra
    Real<int> r1_int(-3); // Using a negative value for demonstration
    Real<int> r2_int(1);

    std::cout << "Integer Real r1 + r2 = " << r1_int + r2_int << std::endl;

    Real<float> r1_float(-3.5); // Using a negative value for demonstration
    Real<float> r2_float(1.2);

    std::cout << "Float Real r1 + r2 = " << r1_float + r2_float << std::endl;

    Real<float> r1(-3.0); // Using a negative value for demonstration
    Real<float> r2(1.0);

    // Basic Arithmetic Operations
    std::cout << "r1 + r2 = " << r1 + r2 << std::endl;
    std::cout << "r1 - r2 = " << r1 - r2 << std::endl;
    std::cout << "r1 * r2 = " << r1 * r2 << std::endl;
    std::cout << "r1 / r2 = " << r1 / r2 << std::endl;

    // Absolute value of Real Numbers
    std::cout << "Absolute value of r1 = " << Real<float>::abs(r1) << std::endl;
    std::cout << "Absolute value of r2 = " << Real<float>::abs(r2) << std::endl;
    std::cout << std::endl;

    // ! Real numbers. Gradients
    // Define x and y with initial values
    Real<double> x(2.0); // Let's say x = 2.0
    Real<double> y(3.0); // Let's say y = 3.0

    // // Define the function f(x, y) = x + y
    // Real<double> f = x + y;
    // std::cout << "f = x + y" << std::endl;

    // // Use the handler to perform the backward pass
    // f.computeGradients();

    // std::cout << "Gradient w.r.t x: " << x.getGradient() << std::endl; // Should be ∂f/∂x
    // std::cout << "Gradient w.r.t y: " << y.getGradient() << std::endl; // Should be ∂f/∂y

    // Define the function f(x, y) = x^2 * y + y + 1
    Real<double> g = x * x * y + y + 1;
    std::cout << "f(x, y) = x^2 * y + y + 1" << std::endl;

    // Use the handler to perform the backward pass
    g.computeGradients();

    // Output the computed gradients using the getters
    std::cout << "Gradient w.r.t x: " << x.getGradient() << std::endl; // Should be ∂f/∂x
    std::cout << "Gradient w.r.t y: " << y.getGradient() << std::endl; // Should be ∂f/∂y
    std::cout << std::endl;

    // ! Complex number algebra
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
    std::cout << "Magnitude of c1 = " << Complex<float>::abs(c1) << std::endl;
    std::cout << "Magnitude of c2 = " << Complex<float>::abs(c2) << std::endl;

    // Conjugate of a Complex Number
    std::cout << "Conjugate of c1 = " << Complex<float>::conjugate(c1) << std::endl;
    std::cout << "Conjugate of c2 = " << Complex<float>::conjugate(c2) << std::endl;
    std::cout << std::endl;

    // ! Matrix algebra
    // Defining a matrix
    Matrix<Complex<float>> matrix(2, 3);
    matrix[{0, 0}] = Complex<float>(1.0, 2.0);
    matrix[{0, 1}] = Complex<float>(3.0, 4.0);
    matrix[{1, 0}] = Complex<float>(5.0, 6.0);
    matrix[{1, 1}] = Complex<float>(7.0, 8.0);

    // Printing the matrix
    std::cout << "Matrix Contents:" << std::endl;
    std::cout << matrix; // Using the overloaded << operator

    Matrix<Complex<float>> negatedMatrix = -matrix;

    std::cout << "Original Matrix:" << std::endl
              << matrix;
    std::cout << "Negated Matrix:" << std::endl
              << negatedMatrix;

    Matrix<Complex<float>> matrix1(2, 2);
    matrix1[{0, 0}] = Complex<float>(1.0, 2.0);
    matrix1[{0, 1}] = Complex<float>(3.0, 4.0);
    matrix1[{1, 0}] = Complex<float>(5.0, 6.0);
    matrix1[{1, 1}] = Complex<float>(7.0, 8.0);

    Matrix<Complex<float>> matrix2(2, 2);
    matrix2[{0, 0}] = Complex<float>(1.0, 1.0);
    matrix2[{0, 1}] = Complex<float>(1.0, 1.0);
    matrix2[{1, 0}] = Complex<float>(1.0, 1.0);
    matrix2[{1, 1}] = Complex<float>(1.0, 1.0);

    Matrix<Complex<float>> sum = matrix1 + matrix2;
    Matrix<Complex<float>> difference = matrix1 - matrix2;
    Matrix<Complex<float>> matmul = Matrix<Complex<float>>::matmul(matrix1, matrix2);

    std::cout << "Sum of Matrices:" << std::endl
              << sum;
    std::cout << "Difference of Matrices:" << std::endl
              << difference;
    std::cout << "Product of Matrices:" << std::endl
              << matmul;

    std::cout << std::endl;

    return 0;
}
