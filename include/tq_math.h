#ifndef _TQ_MATH_H_
#define _TQ_MATH_H_

/**
 * Constants
 */

#define PI 3.141592653589793
#define TWO_PI 6.283185307179586

/**
 * Precision limits.
 */

#define RNG_EPSILON (float)1e-6

/**
 * @brief Draw a value from a random uniform
 * distribution on [min, max).
 *
 * @param min Lower bound.
 * @param max Upper bound.
 * @return float A sample from the distribution.
 */
float TQ_rand_unif(float min, float max);

/**
 * @brief Draw a value from a random normal
 * distribution centered at "mu", with std "sigma".
 * 
 * Use the standard Box–Muller transform.
 *
 * @param mu The real mean of the distribution.
 * @param sigma The real std of the distribution.
 * @return float A sample from the distribution.
 */
float TQ_rand_norm(float mu, float sigma);

#endif