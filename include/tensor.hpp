#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <stack>
#include <iomanip>

template <typename T>
class Tensor
{
public:
    Tensor(const std::vector<size_t> &dims) : m_dims(dims), m_elemsize(1)
    {
        for (size_t dim : dims)
        {
            m_elemsize *= dim;
        }
        m_data.resize(m_elemsize);
    }

    const std::vector<size_t> &shape() const
    {
        return m_dims;
    }

    // Non-const version of operator[]
    T &operator[](const std::vector<size_t> &indices)
    {
        size_t linearIndex = calculateLinearIndex(indices);
        return m_data[linearIndex];
    }

    // Const version of operator[]
    const T &operator[](const std::vector<size_t> &indices) const
    {
        size_t linearIndex = calculateLinearIndex(indices);
        return m_data[linearIndex];
    }

    // Suma de tensores
    Tensor operator+(const Tensor &other) const
    {
        checkDimensionEquality(other);
        Tensor result(m_dims);
        for (size_t i = 0; i < m_elemsize; ++i)
        {
            result.m_data[i] = m_data[i] + other.m_data[i];
        }
        return result;
    }

    // Resta de tensores
    Tensor operator-(const Tensor &other) const
    {
        checkDimensionEquality(other);
        Tensor result(m_dims);
        for (size_t i = 0; i < m_elemsize; ++i)
        {
            result.m_data[i] = m_data[i] - other.m_data[i];
        }
        return result;
    }

    // Producto por un escalar
    Tensor operator*(const T &scalar) const
    {
        Tensor result(m_dims);
        for (size_t i = 0; i < m_elemsize; ++i)
        {
            result.m_data[i] = m_data[i] * scalar;
        }
        return result;
    }

    friend std::ostream &operator<<(std::ostream &out, const Tensor &tensor)
    {
        tensor.print(out);
        return out;
    }

private:
    std::vector<T> m_data;
    std::vector<size_t> m_dims;
    size_t m_elemsize;

    void checkDimensionEquality(const Tensor &other) const
    {
        if (m_dims != other.m_dims)
        {
            throw std::invalid_argument("Tensor dimensions must be equal for arithmetic operations");
        }
    }

    void print(std::ostream &out) const
    {
        if (m_dims.empty())
        {
            out << "[]" << std::endl;
            return;
        }

        std::stack<std::pair<size_t, size_t>> indexStack; // (dimensión, índice en esa dimensión)
        indexStack.push({0, 0});

        while (!indexStack.empty())
        {
            const auto &[dim, idx] = indexStack.top();

            if (dim == m_dims.size())
            {
                out << m_data[calculateLinearIndex(indexStack)];
                closeBracketsAndAdvance(out, indexStack);
            }
            else
            {
                if (idx == 0)
                {
                    out << "["; // Un espacio solo entre dimensiones, no entre corchetes
                }
                pushNextDimensionOrPop(indexStack, dim, idx);
            }
        }
        out << std::endl;
    }

    void closeBracketsAndAdvance(std::ostream &out, std::stack<std::pair<size_t, size_t>> &indexStack) const
    {
        indexStack.pop();
        if (!indexStack.empty())
        {
            indexStack.top().second++;
            if (indexStack.top().second < m_dims[indexStack.top().first])
            {
                out << ", ";
            }
            else
            {
                closeRemainingBrackets(out, indexStack);
            }
        }
    }

    void closeRemainingBrackets(std::ostream &out, std::stack<std::pair<size_t, size_t>> &indexStack) const
    {
        while (!indexStack.empty() && indexStack.top().second >= m_dims[indexStack.top().first])
        {
            out << "]";
            indexStack.pop();
            if (!indexStack.empty())
            {
                indexStack.top().second++;
                if (indexStack.top().second < m_dims[indexStack.top().first])
                {
                    out << ",\n"
                        << std::string(indexStack.top().first > 0 ? 1 : 0, ' '); // Un espacio solo entre dimensiones
                }
            }
        }
    }

    void pushNextDimensionOrPop(std::stack<std::pair<size_t, size_t>> &indexStack, size_t dim, size_t idx) const
    {
        if (idx < m_dims[dim])
        {
            indexStack.push({dim + 1, 0});
        }
        else
        {
            indexStack.pop();
            if (!indexStack.empty())
            {
                indexStack.top().second++;
            }
        }
    }

    // Calcular índice lineal basado en la pila de índices
    size_t calculateLinearIndex(const std::stack<std::pair<size_t, size_t>> &indexStack) const
    {
        size_t linearIndex = 0;
        size_t accumulatedProduct = 1;

        std::stack<std::pair<size_t, size_t>> tempStack(indexStack);
        while (!tempStack.empty())
        {
            const auto &[dim, idx] = tempStack.top();
            tempStack.pop();

            if (!tempStack.empty())
            {
                linearIndex += idx * accumulatedProduct;
                accumulatedProduct *= m_dims[dim];
            }
        }

        return linearIndex;
    }

    size_t calculateLinearIndex(const std::vector<size_t> &indices) const
    {
        if (indices.size() != m_dims.size())
        {
            throw std::invalid_argument("Index dimension mismatch");
        }

        size_t linearIndex = 0;
        size_t accumulatedProduct = 1;
        for (int i = indices.size() - 1; i >= 0; --i)
        {
            if (indices[i] >= m_dims[i])
            {
                throw std::out_of_range("Index out of range");
            }
            linearIndex += indices[i] * accumulatedProduct;
            accumulatedProduct *= m_dims[i];
        }
        return linearIndex;
    }
};

#endif // TENSOR_HPP
