#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../include/tq_math.h"

/**
 * @brief Initialise C random number generator.
 *
 */
void TQ_Init_rand()
{
    static char is_rand_init = 0;

    if (!is_rand_init)
    {
        srand((unsigned int)time(NULL));
        is_rand_init = 1;
    }
}

float TQ_rand_unif(float min, float max)
{
    float range;
    float rand_val;

    TQ_Init_rand();

    range = max - min;
    rand_val = ((float)rand() / (float)RAND_MAX);
    rand_val = range * rand_val + min;

    return rand_val;
}

float TQ_rand_norm(float mu, float sigma)
{
    static float u1 = 0.0;
    float u2;

    float mag;
    float z0; // This is the value we will return.
    // float z1;

    TQ_Init_rand();

    // Only get a rand. unif. value less than EPS
    // once.
    while (u1 <= RNG_EPSILON)
    {
        u1 = (float)rand() / (float)RAND_MAX;
    }

    // Get another value from the unif.
    u2 = (float)rand() / (float)RAND_MAX;

    // Compute rand. norm. sample N(mu, sigma):
    // z0, z1
    mag = sigma * sqrtf(-2.0 * logf(u1));
    z0 = mag * cosf(TWO_PI * u2) + mu;
    // z1  = mag * sinf(TWO_PI * u2) + mu;

    return z0;
}