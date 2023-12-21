#ifndef __REAL_HPP_
#define __REAL_HPP_

#include <iostream>
#include <stdexcept>
#include <memory>
#include <cmath> // For std::abs

template <typename T>
class Real
{
public:
    // Constructor modified to create a node with a no-op backward function
    Real(T value = 0) : m_value(value) {}

    T value() const
    {
        return m_value;
    }

    // Arithmetic Operators
    Real operator+(const Real &other) const
    {
        T resultValue = m_value + other.m_value;
        return Real(resultValue);
    }

    Real operator-(const Real &other) const
    {
        T resultValue = m_value - other.m_value;
        return Real(resultValue);
    }

    Real operator*(const Real &other) const
    {
        T resultValue = m_value * other.m_value;
        return Real(resultValue);
    }

    Real operator/(const Real &other) const
    {
        if (other.m_value == 0)
        {
            throw std::runtime_error("Division by zero in real division");
        }

        T resultValue = m_value / other.m_value;

        return Real(resultValue);
    }

    // Compound Assignment Operators
    Real &operator+=(const Real &rhs)
    {
        *this = *this + rhs;
        return *this;
    }

    Real &operator-=(const Real &rhs)
    {
        *this = *this - rhs;
        return *this;
    }

    Real &operator*=(const Real &rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    Real &operator/=(const Real &rhs)
    {
        *this = *this / rhs;
        return *this;
    }

    // Overloaded operators for Real + T, Real - T, etc.
    Real operator+(T other) const
    {
        return *this + Real(other); // Reuse the existing Real + Real operator
    }

    Real operator-(T other) const
    {
        return *this - Real(other); // Reuse the existing Real - Real operator
    }

    Real operator*(T other) const
    {
        return *this * Real(other); // Reuse the existing Real * Real operator
    }

    Real operator/(T other) const
    {
        return *this / Real(other); // Reuse the existing Real / Real operator
    }

    // Unary Operators (Real<T> op= T) using existing binary operators
    Real &operator+=(T other)
    {
        *this = *this + Real(other); // Using existing operator+
        return *this;
    }

    Real &operator-=(T other)
    {
        *this = *this - Real(other); // Using existing operator-
        return *this;
    }

    Real &operator*=(T other)
    {
        *this = *this * Real(other); // Using existing operator*
        return *this;
    }

    Real &operator/=(T other)
    {
        *this = *this / Real(other); // Using existing operator/
        return *this;
    }

    // Absolute Value Function
    Real abs() const
    {
        T resultValue = std::abs(m_value);
        return Real(resultValue);
    }

    static Real abs(const Real &r)
    {
        return r.abs();
    }

    friend std::ostream &operator<<(std::ostream &out, const Real<T> &real)
    {
        out << real.m_value;
        return out;
    }

private:
    T m_value;
};

template <typename T>
Real<T> operator+(T lhs, const Real<T> &rhs)
{
    return Real<T>(lhs) + rhs; // Reuse the existing Real + Real operator
}

template <typename T>
Real<T> operator-(T lhs, const Real<T> &rhs)
{
    return Real<T>(lhs) - rhs; // Reuse the existing Real - Real operator
}

template <typename T>
Real<T> operator*(T lhs, const Real<T> &rhs)
{
    return Real<T>(lhs) * rhs; // Reuse the existing Real * Real operator
}

template <typename T>
Real<T> operator/(T lhs, const Real<T> &rhs)
{
    return Real<T>(lhs) / rhs; // Reuse the existing Real / Real operator
}

#endif // __REAL_HPP_
