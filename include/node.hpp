#ifndef __NODE_HPP_
#define __NODE_HPP_

#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>

template <typename T>
class Node : public std::enable_shared_from_this<Node<T>> // Specify the template argument here
{
public:
    Node(T val, std::function<void()> backwardOp)
        : m_value(val), m_gradient(0.0), m_backward_f(backwardOp) {}

    void addParent(std::shared_ptr<Node<T>> parent)
    {
        m_parent_vector.push_back(parent);
    }

    void computeGradientsTopologically()
    {
        std::vector<std::shared_ptr<Node<T>>> Stack;
        std::unordered_set<std::shared_ptr<Node<T>>> visited;

        // Call the recursive helper function to store Topological Sort
        // starting from all vertices one by one
        topologicalSortUtil(shared_from_this(), visited, Stack);

        // Once sorted, compute gradients in reverse topological order
        while (!Stack.empty())
        {
            auto node = Stack.back();
            Stack.pop_back();
            node->m_backward_f();
        }
    }

    T m_value;
    T m_gradient;

private:
    void topologicalSortUtil(std::shared_ptr<Node<T>> node,
                             std::unordered_set<std::shared_ptr<Node<T>>> &visited,
                             std::vector<std::shared_ptr<Node<T>>> &Stack)
    {
        // Mark the current node as visited
        visited.insert(node);

        // Recur for all the vertices adjacent to this vertex
        for (auto &parent : node->m_parent_vector)
        {
            auto parentPtr = parent.lock();
            if (parentPtr && visited.find(parentPtr) == visited.end())
            {
                topologicalSortUtil(parentPtr, visited, Stack);
            }
        }

        // Push current vertex to stack which stores result
        Stack.push_back(node);
    }

    std::vector<std::weak_ptr<Node>> m_parent_vector; // Using weak_ptr to avoid circular references
    std::function<void()> m_backward_f;
};

#endif // __NODE_HPP_
