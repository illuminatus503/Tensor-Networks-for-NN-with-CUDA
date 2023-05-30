#ifndef _TQ_NEURAL_H_
#define _TQ_NEURAL_H_

#include "tq_matrix.h"

typedef struct TQ_Activation
{
    float (*activation)(float);       // Activation function
    float (*deriv_activation)(float); // Deriv. activation function
} TQ_Activation;

/**
 * @brief A Logistic regresion object.
 * Multiple input, but a single output (given by a sigmoid function).
 */
typedef struct TQ_Perceptron
{
    TQ_Tensor weights_vector; // Weight vector (saved transposed)
    TQ_Tensor ext_input;      // Extended INPUT (X) value
    TQ_Tensor transfer_f;     // Z value of the perceptron
    TQ_Tensor activation_v;   // A value of the perceptron
    TQ_Tensor dW;             // delJ / delW (delta)
    TQ_Activation activ;      // Activation funcion (and deriv.)
    TQ_Tensor_type type;      // Perceptron Type: CPU or GPU?
} TQ_Perceptron;

void TQ_Perceptron_Create(TQ_Perceptron *neuron,
                          unsigned int num_features,
                          TQ_Tensor_type type);

void TQ_Perceptron_Free(TQ_Perceptron *neuron);

void TQ_Perceptron_Forward(TQ_Tensor X,
                           TQ_Perceptron *neuron);
void TQ_Perceptron_Backward(TQ_Perceptron *neuron,
                            TQ_Tensor Y);

#endif