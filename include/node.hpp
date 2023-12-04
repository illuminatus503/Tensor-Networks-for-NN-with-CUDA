#ifndef __NODE_HPP_
#define __NODE_HPP_

#include <memory>
#include <vector>
#include <functional>

struct Node
{
    double value;
    double gradient;

    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> backward;

    Node(double val) : value(val), gradient(0) {}

    void computeGradients()
    {
        backward();
        
        for (auto &parent : parents)
        {
            parent->computeGradients();
        }
    }
};

#endif // __NODE_HPP_