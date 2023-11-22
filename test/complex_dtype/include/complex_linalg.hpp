#ifndef __COMPLEX_LINALG_HPP_
#define __COMPLEX_LINALG_HPP_

#include "complex.hpp"
#include <cmath> // For sqrt

template <typename T>
T magnitude(const Complex<T> &c)
{
    return std::sqrt(c.real() * c.real() + c.imag() * c.imag());
}

template <typename T>
Complex<T> conjugate(const Complex<T> &c)
{
    return Complex<T>(c.real(), -c.imag());
}

#endif // __COMPLEX_LINALG_HPP_