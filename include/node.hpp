#ifndef __NODE_HPP_
#define __NODE_HPP_

#include <memory>
#include <vector>
#include <functional>

template <typename T>
struct Node
{
public:
    Node(T val, std::function<void()> backwardOp) : m_value(val), m_gradient(0.0), m_backward_f(backwardOp) {}

    void computeGradients()
    {
        // (1) Update my gradient
        this->m_backward_f();

        // (2) Update the gradients of my parents, based on my gradient
        for (auto &weak_parent : this->m_parent_vector)
        {
            // Lock the weak_ptr to get a shared_ptr
            if (auto parent = weak_parent.lock())
            {
                parent->computeGradients();
            }
        }
    }

    T m_value;
    T m_gradient;

    std::vector<std::weak_ptr<Node>> m_parent_vector; // Using weak_ptr to avoid circular references
    std::function<void()> m_backward_f;
};

#endif // __NODE_HPP_
