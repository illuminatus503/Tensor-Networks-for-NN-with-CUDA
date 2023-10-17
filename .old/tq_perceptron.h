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
    struct TQ_Matrix weights_vector; // Weight vector (saved transposed)
    struct TQ_Matrix ext_input;      // Extended INPUT (X) value
    struct TQ_Matrix transfer_f;     // Z value of the perceptron
    struct TQ_Matrix activation_v;   // A value of the perceptron
    struct TQ_Matrix dW;             // delJ / delW (delta)
    struct TQ_Activation activ;      // Activation funcion (and deriv.)
    enum TQ_Matrix_type type;        // Perceptron Type: CPU or GPU?
} typedef TQ_Perceptron;

void TQ_Perceptron_Create(struct TQ_Perceptron *neuron,
                          unsigned int num_features,
                          enum TQ_Matrix_type type);

void TQ_Perceptron_Free(struct TQ_Perceptron *neuron);

void TQ_Perceptron_Forward(struct TQ_Matrix X,
                           struct TQ_Perceptron *neuron);
void TQ_Perceptron_Backward(struct TQ_Perceptron *neuron,
                            struct TQ_Matrix Y);

#endif