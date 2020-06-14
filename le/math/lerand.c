#include "lerand.h"
#include <stdlib.h>
#include <math.h>

float
le_random_uniform_f32(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float
le_random_normal_f32(void)
{
    float u1 = le_random_uniform_f32();
    float u2 = le_random_uniform_f32();
    return cosf(2.0f * (float)M_PI * u2) * sqrtf(-2.0f * logf(u1));
}
