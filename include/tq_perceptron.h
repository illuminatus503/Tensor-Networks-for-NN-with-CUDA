#ifndef _TQ_NEURAL_H_
#define _TQ_NEURAL_H_

#include "tq_matrix.h"

/**
 * @brief A Logistic regresion object.
 * Multiple input, but a single output (given by a sigmoid function).
 */
struct TQ_Perceptron
{
    struct TQ_Matrix weights_vector; // Weight vector
    float (*activation)(float);      // Activation function
} typedef TQ_Perceptron;

void TQ_Perceptron_Create(struct TQ_Perceptron *neuron,
                          unsigned int num_input);

#endif