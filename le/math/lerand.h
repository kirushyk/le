#ifndef __LERAND_H__
#define __LERAND_H__

#include <le/lemacros.h>

LE_BEGIN_DECLS

float                   le_random_uniform_f32              (void);

float                   le_random_normal_f32               (void);

typedef enum LeDistribution
{
    LE_DISTRIBUTION_UNIFORM,
    LE_DISTRIBUTION_NORMAL
} LeDistribution;

float                   le_random_f32                      (LeDistribution          distribution);

LE_END_DECLS

#endif
