#ifndef _TQ_NEURAL_H_
#define _TQ_NEURAL_H_

#include "tq_matrix.h"

struct TQ_Activation
{
    float (*activation)(float);       // Activation function
    float (*deriv_activation)(float); // Deriv. activation function
} typedef TQ_Activation;

/**
 * @brief A Logistic regresion object.
 * Multiple input, but a single output (given by a sigmoid function).
 */
struct TQ_Perceptron
{
    struct TQ_Matrix weights_vector; // Weight vector
    struct TQ_Activation activ;      // Activation funcion (and deriv.)
    enum TQ_Matrix_type type;        // Perceptron Type: CPU or GPU?
} typedef TQ_Perceptron;

void TQ_Perceptron_Create(struct TQ_Perceptron *neuron,
                          unsigned int num_input,
                          enum TQ_Matrix_type type);

void TQ_Perceptron_Forward(struct TQ_Matrix X,
                           struct TQ_Perceptron neuron,
                           struct TQ_Matrix *A);

#endif