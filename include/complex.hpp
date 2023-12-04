#ifndef __COMPLEX_HPP_
#define __COMPLEX_HPP_

#include <iostream>
#include <stdexcept>
#include <cmath> // For sqrt

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
        T denominator = sq_abs(other);
        T newReal = (m_real * other.m_real + m_imag * other.m_imag) / denominator;
        T newImag = (m_imag * other.m_real - m_real * other.m_imag) / denominator;
        return Complex(newReal, newImag);
    }

    // Overload the += operator
    Complex &operator+=(const Complex &rhs)
    {
        this->m_real += rhs.m_real; // Assume 'real' and 'imag' are members of Complex
        this->m_imag += rhs.m_imag;
        return *this;
    }

    // Overload the -= operator
    Complex &operator-=(const Complex &rhs)
    {
        m_real -= rhs.m_real;
        m_imag -= rhs.m_imag;
        return *this;
    }

    // Overload the *= operator
    Complex &operator*=(const Complex &rhs)
    {
        float new_real = m_real * rhs.m_real - m_imag * rhs.m_imag;
        float new_imag = m_real * rhs.m_imag + m_imag * rhs.m_real;
        m_real = new_real;
        m_imag = new_imag;
        return *this;
    }

    // Overload the /= operator
    Complex &operator/=(const Complex &rhs)
    {
        if (rhs.m_real == 0 && rhs.m_imag == 0)
        {
            throw std::runtime_error("Division by zero in complex division");
        }

        float denominator = sq_abs(rhs);
        float new_real = (m_real * rhs.m_real + m_imag * rhs.m_imag) / denominator;
        float new_imag = (m_imag * rhs.m_real - m_real * rhs.m_imag) / denominator;
        m_real = new_real;
        m_imag = new_imag;
        return *this;
    }

    // Complex conjugate
    static Complex conjugate(const Complex &c)
    {
        return Complex(c.m_real, -c.m_imag);
    }

    // The magnitude squearred of a complex number
    static double sq_abs(const Complex &c)
    {
        return c.m_real * c.m_real + c.m_imag * c.m_imag;
    }

    // The magnitude of a complex number
    static double abs(const Complex &c)
    {
        return std::sqrt(sq_abs(c));
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
