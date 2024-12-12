#include "lerand.h"
#include <stdlib.h>
#include <math.h>
#include <glib.h>

gfloat
le_random_uniform_f32(void)
{
    return (gfloat)rand() / (gfloat)RAND_MAX;
}

gfloat
le_random_normal_f32(void)
{
    gfloat u1 = le_random_uniform_f32();
    gfloat u2 = le_random_uniform_f32();
    return cosf(2.0f * (gfloat)G_PI * u2) * sqrtf(-2.0f * logf(u1));
}

gfloat
le_random_f32(LeDistribution distribution)
{
    switch (distribution)
    {
    case LE_DISTRIBUTION_NORMAL:
        return le_random_normal_f32();
    
    case LE_DISTRIBUTION_UNIFORM:
    default:
        return le_random_uniform_f32();
    }
}
