#ifndef __REAL_HPP_
#define __REAL_HPP_

#include <iostream>
#include <stdexcept>

#include "node.hpp"

template <typename T>
class Real
{
public:
    Real(T value = 0) : m_value(value) {}

    T value() const
    {
        return m_value;
    }

    Real operator+(const Real &other) const
    {
        return Real(m_value + other.m_value);
    }

    Real operator-(const Real &other) const
    {
        return Real(m_value - other.m_value);
    }

    Real operator-() const
    {
        return Real(-m_value);
    }

    Real operator*(const Real &other) const
    {
        return Real(m_value * other.m_value);
    }

    Real operator/(const Real &other) const
    {
        if (other.m_value == 0)
        {
            throw std::runtime_error("Division by zero in real division");
        }
        return Real(m_value / other.m_value);
    }

    // Overload the += operator
    Real &operator+=(const Real &rhs)
    {
        this->m_value += rhs.m_value;
        return *this;
    }

    // Overload the -= operator
    Real &operator-=(const Real &rhs)
    {
        m_value -= rhs.m_value;
        return *this;
    }

    // Overload the *= operator
    Real &operator*=(const Real &rhs)
    {
        m_value *= rhs.m_value;
        return *this;
    }

    // Overload the /= operator
    Real &operator/=(const Real &rhs)
    {
        if (rhs.m_value == 0)
        {
            throw std::runtime_error("Division by zero in real division");
        }
        m_value /= rhs.m_value;
        return *this;
    }

    // Static method for absolute value
    static Real abs(const Real &r)
    {
        return Real(std::abs(r.m_value));
    }

    friend std::ostream &operator<<(std::ostream &out, const Real<T> &real)
    {
        out << real.m_value;
        return out;
    }

private:
    T m_value;
    std::shared_ptr<Node> node;
};

#endif // __REAL_HPP_
