// Tensor.hpp
#ifndef __TENSOR_HPP_
#define __TENSOR_HPP_

#include <iostream>
#include <vector>

template <typename T>
class Tensor
{
public:
    Tensor(size_t m_rows, size_t m_cols) : m_rows(m_rows), m_cols(m_cols)
    {
        m_data.resize(m_rows, std::vector<T>(m_cols));
    }

    void setElement(size_t row, size_t col, const T &value)
    {
        if (row < m_rows && col < m_cols)
        {
            m_data[row][col] = value;
        }
    }

    T getElement(size_t row, size_t col) const
    {
        if (row < m_rows && col < m_cols)
        {
            return m_data[row][col];
        }
        return nullptr;
    }

    Tensor operator+(const Tensor &other) const
    {
        Tensor result(m_rows, m_cols);

        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
            {
                result.setElement(i, j, m_data[i][j] + other.m_data[i][j]);
            }
        }

        return result;
    }

    Tensor operator-() const
    {
        Tensor result(m_rows, m_cols);

        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
            {
                result.setElement(i, j, -m_data[i][j]);
            }
        }

        return result;
    }

    Tensor operator-(const Tensor &other) const
    {
        Tensor result(m_rows, m_cols);
        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
            {
                result.setElement(i, j, m_data[i][j] - other.m_data[i][j]);
            }
        }
        return result;
    }

    friend std::ostream &operator<<(std::ostream &out, const Tensor<T> &tensor)
    {
        for (size_t i = 0; i < tensor.m_rows; ++i)
        {
            for (size_t j = 0; j < tensor.m_cols; ++j)
            {
                out << tensor.m_data[i][j] << "\t";
            }
            out << std::endl;
        }
        return out;
    }

private:
    std::vector<std::vector<T>> m_data;
    size_t m_rows, m_cols;
};

#endif // __TENSOR_HPP_
