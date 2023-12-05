#ifndef __REAL_HPP_
#define __REAL_HPP_

#include <iostream>
#include <stdexcept>
#include <memory>
#include <cmath> // For std::abs
#include "node.hpp"

template <typename T>
class Real
{
public:
    // Constructor modified to create a node with a no-op backward function
    Real(T value = 0) : m_value(value), m_node(std::make_shared<Node<T>>(value, []() {})) {}

    T value() const { return m_value; }

    // Arithmetic Operators
    Real operator+(const Real &other) const
    {
        T resultValue = m_value + other.m_value;
        auto backwardOp = [this, other]()
        {
            this->m_node->m_gradient += 1;
            other.m_node->m_gradient += 1;
        };
        return Real(resultValue, createNodeForOperation(other, resultValue, backwardOp));
    }

    Real operator-(const Real &other) const
    {
        T resultValue = m_value - other.m_value;
        auto backwardOp = [this, other]()
        {
            this->m_node->m_gradient += 1;
            other.m_node->m_gradient -= 1;
        };
        return Real(resultValue, createNodeForOperation(other, resultValue, backwardOp));
    }

    Real operator*(const Real &other) const
    {
        T resultValue = m_value * other.m_value;
        auto backwardOp = [this, other]()
        {
            this->m_node->m_gradient += other.m_value;
            other.m_node->m_gradient += this->m_value;
        };
        return Real(resultValue, createNodeForOperation(other, resultValue, backwardOp));
    }

    Real operator/(const Real &other) const
    {
        if (other.m_value == 0)
        {
            throw std::runtime_error("Division by zero in real division");
        }

        T resultValue = m_value / other.m_value;
        auto backwardOp = [this, other]()
        {
            this->m_node->m_gradient += 1 / other.m_value;
            other.m_node->m_gradient -= this->m_value / (other.m_value * other.m_value);
        };

        return Real(resultValue, createNodeForOperation(other, resultValue, backwardOp));
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

    // Absolute Value Function
    Real abs() const
    {
        T resultValue = std::abs(m_value);
        auto backwardOp = [this]()
        {
            this->m_node->m_gradient += (this->m_value >= 0 ? 1.0 : -1.0);
        };
        return Real(resultValue, createNodeForOperation(resultValue, backwardOp));
    }

    static Real abs(const Real &r)
    {
        return r.abs();
    }

    // Gradient Computation
    void computeGradients()
    {
        if (m_node)
        {
            m_node->m_gradient = 1.0; // Seed the gradient
            m_node->computeGradients();
        }
    }

    // Getter for Gradient
    T getGradient() const
    {
        if (m_node)
        {
            return m_node->m_gradient;
        }
        else
        {
            return 0.0;
        }
    }

    friend std::ostream &operator<<(std::ostream &out, const Real<T> &real)
    {
        out << real.m_value;
        return out;
    }

private:
    T m_value;
    std::shared_ptr<Node<T>> m_node;

    // Private Constructor for Internal Use
    Real(T value, std::shared_ptr<Node<T>> node) : m_value(value), m_node(node) {}

    // Helper method to create a new node for an operation
    std::shared_ptr<Node<T>> createNodeForOperation(const Real &other, T resultValue, std::function<void()> backwardOp) const
    {
        auto newNode = std::make_shared<Node<T>>(resultValue, backwardOp);
        newNode->m_parent_vector.push_back(this->m_node);
        newNode->m_parent_vector.push_back(other.m_node);
        return newNode;
    }

    // Helper method to create a new node for an operation
    std::shared_ptr<Node<T>> createNodeForOperation(T resultValue, std::function<void()> backwardOp) const
    {
        auto newNode = std::make_shared<Node<T>>(resultValue, backwardOp);
        newNode->m_parent_vector.push_back(this->m_node);
        return newNode;
    }
};

#endif // __REAL_HPP_
