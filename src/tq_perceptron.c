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

void TQ_Perceptron_Create(TQ_Perceptron *neuron,
                          unsigned int num_features,
                          TQ_Matrix_type type)
{
    unsigned int dims[2];
    dims[0] = num_features + 1; // Weights + Bias
    dims[1] = 1;

    // Creamos el vector de pesos: vector random.
    TQ_Matrix_Create(&(neuron->weights_vector), dims, 2, type);
    TQ_Matrix_Unif(&(neuron->weights_vector));
    // Existen placeholder para Z, A que se inicializan una vez hecho
    // el forward pass.

    // Guardamos el tipo del perceptron: CPU o GPU
    neuron->type = type;

    // Guardamos la función de activación: sigmoide & deriv. sigmoide
    neuron->activ.activation = &sigmoid;
    neuron->activ.deriv_activation = &dsigmoid;
}

void TQ_Perceptron_Free(TQ_Perceptron *neuron)
{
    TQ_Matrix_Free(&(neuron->weights_vector));

    if (neuron->ext_input.h_mem != NULL)
    {
        TQ_Matrix_Free(&(neuron->ext_input));
    }

    if (neuron->transfer_f.h_mem != NULL)
    {
        TQ_Matrix_Free(&(neuron->transfer_f));
    }

    if (neuron->activation_v.h_mem != NULL)
    {
        TQ_Matrix_Free(&(neuron->activation_v));
    }

    if (neuron->dW.h_mem != NULL)
    {
        TQ_Matrix_Free(&(neuron->dW));
    }
}

void TQ_Perceptron_Forward(TQ_Matrix X,
                           TQ_Perceptron *neuron)
{
    // Extiende la entrada por 1 en la dimensión 0 (filas)
    unsigned int ext_dims[2] = {X.dimensions[0], X.dimensions[1] + 1};
    TQ_Matrix_Extend(X, &(neuron->ext_input), ext_dims, 2, 1.0f);

    // Transference function (neuron.weights_vector = W, by def.)
    TQ_Matrix_Prod(neuron->ext_input, neuron->weights_vector, &(neuron->transfer_f));

    // Activation
    TQ_Matrix_Apply(neuron->transfer_f, neuron->activ.activation, &(neuron->activation_v));
}

void TQ_Perceptron_Backward(TQ_Perceptron *neuron,
                            TQ_Matrix Y)
{
    TQ_Matrix dZ; // delJ/delZ equiv.
    TQ_Matrix X_t;

    // dZ (Y.sizeof(i), 1)
    TQ_Matrix_Sub(neuron->activation_v, Y, &dZ);

    // dW (Y.sizeof(i) + 1, 1) Incluye el bias
    TQ_Matrix_T(neuron->ext_input, &X_t);
    TQ_Matrix_Prod(X_t, dZ, &(neuron->dW));

    TQ_Matrix_Free(&dZ);
    TQ_Matrix_Free(&X_t);
}