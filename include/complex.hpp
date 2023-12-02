#ifndef __COMPLEX_HPP_
#define __COMPLEX_HPP_

#include <iostream>
#include <cmath> // For sqrt

/**
 * The usual approach is to define the entire template class,
 * including its methods, in the header file. This separation
 * technique is a workaround and should be used with understan-
 * ding of its limitations.
 */

template <typename T>
class Complex
{
public:
    Complex(T r = 0, T i = 0) : m_real(r), m_imag(i) {}

    T real() const
    {
        return m_real;
    }

    T imag() const
    {
        return m_imag;
    }

    Complex operator+(const Complex &other) const
    {
        return Complex(m_real + other.m_real, m_imag + other.m_imag);
    }

    Complex operator-(const Complex &other) const
    {
        return Complex(m_real - other.m_real, m_imag - other.m_imag);
    }

    Complex operator-() const
    {
        return Complex(-m_real, -m_imag);
    }

    Complex operator*(const Complex &other) const
    {
        T newReal = m_real * other.m_real - m_imag * other.m_imag;
        T newImag = m_real * other.m_imag + m_imag * other.m_real;
        return Complex(newReal, newImag);
    }

    Complex operator/(const Complex &other) const
    {
        T denominator = other.m_real * other.m_real + other.m_imag * other.m_imag;
        T newReal = (m_real * other.m_real + m_imag * other.m_imag) / denominator;
        T newImag = (m_imag * other.m_real - m_real * other.m_imag) / denominator;
        return Complex(newReal, newImag);
    }

    friend std::ostream &operator<<(std::ostream &out, const Complex<T> &c)
    {
        out << c.m_real;
        if (c.m_imag >= 0)
            out << "+";
        out << c.m_imag << "i";
        return out;
    }

private:
    T m_real;
    T m_imag;
};

#endif // __COMPLEX_HPP_
