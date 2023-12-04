#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <utility> // for std::pair

template <typename T>
class Matrix
{
public:
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols)
    {
        m_data.resize(m_rows * m_cols);
    }

    // Non-const version of operator[]
    T &operator[](const std::pair<size_t, size_t> &indices)
    {
        size_t linearIndex = calculateIndex(indices);
        return m_data[linearIndex];
    }

    // Const version of operator[]
    const T &operator[](const std::pair<size_t, size_t> &indices) const
    {
        size_t linearIndex = calculateIndex(indices);
        return m_data[linearIndex];
    }

    // Matrix addition
    Matrix operator+(const Matrix &other) const
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
        {
            throw std::invalid_argument("Matrix dimensions must be equal for addition");
        }

        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
            {
                result[{i, j}] = (*this)[{i, j}] + other[{i, j}];
            }
        }
        return result;
    }

    // Unary negation
    Matrix operator-() const
    {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
            {
                result[{i, j}] = -(*this)[{i, j}];
            }
        }
        return result;
    }

    // Matrix subtraction
    Matrix operator-(const Matrix &other) const
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
        {
            throw std::invalid_argument("Matrix dimensions must be equal for subtraction");
        }

        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
            {
                result[{i, j}] = (*this)[{i, j}] - other[{i, j}];
            }
        }
        return result;
    }

    // Matrix multiplication: static method
    static Matrix matmul(const Matrix &this_, const Matrix &other)
    {
        if (this_.m_cols != other.m_rows)
        {
            throw std::invalid_argument("Matrix dimensions must be compatible for multiplication");
        }

        Matrix result(this_.m_rows, other.m_cols);

        for (size_t i = 0; i < this_.m_rows; i++)
        {
            for (size_t j = 0; j < other.m_cols; j++)
            {
                for (size_t k = 0; k < this_.m_cols; k++)
                {
                    result[{i, j}] += this_[{i, k}] * other[{k, j}];
                }
            }
        }

        return result;
    }

    // Output stream overload
    friend std::ostream &operator<<(std::ostream &out, const Matrix<T> &matrix)
    {
        for (size_t i = 0; i < matrix.m_rows; ++i)
        {
            for (size_t j = 0; j < matrix.m_cols; ++j)
            {
                out << matrix[{i, j}] << "\t";
            }
            out << std::endl;
        }
        return out;
    }

private:
    std::vector<T> m_data;
    size_t m_rows, m_cols;

    size_t calculateIndex(const std::pair<size_t, size_t> &indices) const
    {
        if (indices.first >= m_rows || indices.second >= m_cols)
        {
            throw std::out_of_range("Index out of range");
        }
        return indices.first * m_cols + indices.second;
    }
};

#endif // MATRIX_HPP
