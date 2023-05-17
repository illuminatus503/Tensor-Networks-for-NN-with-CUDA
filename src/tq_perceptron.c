#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#include "../include/tq_perceptron.h"

float sigmoid(float z)
{
    return (1.0f / (1.0f + expf(-1.0f * z)));
}

float dsigmoid(float z)
{
    float exp_z;

    exp_z = expf(-1.0f * z);
    return (exp_z / (1.0f + exp_z));
}

/**
 * Creating a perceptron.
 */

void TQ_Perceptron_Create(struct TQ_Perceptron *neuron,
                          unsigned int num_input,
                          enum TQ_Matrix_type type)
{
    unsigned int dims[2];
    dims[0] = 1; // Weights + Bias
    dims[1] = num_input + 1;

    // Creamos el vector de pesos: vector random.
    TQ_Matrix_Create(&(neuron->weights_vector), dims, 2, type);
    TQ_Matrix_Unif(&(neuron->weights_vector));

    // Guardamos el tipo del perceptron: CPU o GPU
    neuron->type = type;

    // Guardamos la función de activación: sigmoide & deriv. sigmoide
    neuron->activ.activation = &sigmoid;
    neuron->activ.deriv_activation = &dsigmoid;
}

void TQ_Perceptron_Free(struct TQ_Perceptron *neuron)
{
    TQ_Matrix_Free(&(neuron->weights_vector));
}

/**
 * Forward pass.
 */
void TQ_Perceptron_Forward(struct TQ_Matrix X,
                           struct TQ_Perceptron neuron,
                           struct TQ_Matrix *A)
{
    struct TQ_Matrix Z, X_ext;

    // Transference function (neuron.weights_vector = W_T, by def.)
    TQ_Matrix_Prod(neuron.weights_vector, X, &Z);

    // Activation
    TQ_Matrix_Apply(Z, neuron.activ.activation, A);
    TQ_Matrix_Free(&Z);
}