#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <utility> // for std::pair

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

    /**
     * @brief Return the shape of the tensor. Inmutable.
     * 
     * @return const std::vector<size_t>& A vector of shapes.
     */
    const std::vector<size_t> &shape() const
    {
        return m_dims;
    }

    // Non-const version of operator[]
    // T &operator[](const std::pair<size_t, size_t> &indices)
    // {
    //     size_t linearIndex = calculateIndex(indices);
    //     return m_data[linearIndex];
    // }

    // // Const version of operator[]
    // const T &operator[](const std::pair<size_t, size_t> &indices) const
    // {
    //     size_t linearIndex = calculateIndex(indices);
    //     return m_data[linearIndex];
    // }

    // // Tensor addition
    // Tensor operator+(const Tensor &other) const
    // {
    //     if (m_rows != other.m_rows || m_cols != other.m_cols)
    //     {
    //         throw std::invalid_argument("Tensor dimensions must be equal for addition");
    //     }

    //     Tensor result(m_rows, m_cols);
    //     for (size_t i = 0; i < m_rows; ++i)
    //     {
    //         for (size_t j = 0; j < m_cols; ++j)
    //         {
    //             result[{i, j}] = (*this)[{i, j}] + other[{i, j}];
    //         }
    //     }
    //     return result;
    // }

    // // Unary negation
    // Tensor operator-() const
    // {
    //     Tensor result(m_rows, m_cols);
    //     for (size_t i = 0; i < m_rows; ++i)
    //     {
    //         for (size_t j = 0; j < m_cols; ++j)
    //         {
    //             result[{i, j}] = -(*this)[{i, j}];
    //         }
    //     }
    //     return result;
    // }

    // // Tensor subtraction
    // Tensor operator-(const Tensor &other) const
    // {
    //     if (m_rows != other.m_rows || m_cols != other.m_cols)
    //     {
    //         throw std::invalid_argument("Tensor dimensions must be equal for subtraction");
    //     }

    //     Tensor result(m_rows, m_cols);
    //     for (size_t i = 0; i < m_rows; ++i)
    //     {
    //         for (size_t j = 0; j < m_cols; ++j)
    //         {
    //             result[{i, j}] = (*this)[{i, j}] - other[{i, j}];
    //         }
    //     }
    //     return result;
    // }

    // // Tensor multiplication: static method
    // static Tensor matmul(const Tensor &this_, const Tensor &other)
    // {
    //     if (this_.m_cols != other.m_rows)
    //     {
    //         throw std::invalid_argument("Tensor dimensions must be compatible for multiplication");
    //     }

    //     Tensor result(this_.m_rows, other.m_cols);

    //     for (size_t i = 0; i < this_.m_rows; i++)
    //     {
    //         for (size_t j = 0; j < other.m_cols; j++)
    //         {
    //             for (size_t k = 0; k < this_.m_cols; k++)
    //             {
    //                 result[{i, j}] += this_[{i, k}] * other[{k, j}];
    //             }
    //         }
    //     }

    //     return result;
    // }

    // // Output stream overload
    // friend std::ostream &operator<<(std::ostream &out, const Tensor<T> &matrix)
    // {
    //     for (size_t i = 0; i < matrix.m_rows; ++i)
    //     {
    //         for (size_t j = 0; j < matrix.m_cols; ++j)
    //         {
    //             out << matrix[{i, j}] << "\t";
    //         }
    //         out << std::endl;
    //     }
    //     return out;
    // }

private:
    std::vector<T> m_data;
    std::vector<size_t> m_dims;
    size_t m_elemsize;

    // size_t calculateIndex(const std::pair<size_t, size_t> &indices) const
    // {
    //     if (indices.first >= m_rows || indices.second >= m_cols)
    //     {
    //         throw std::out_of_range("Index out of range");
    //     }
    //     return indices.first * m_cols + indices.second;
    // }
};

#endif // TENSOR_HPP
