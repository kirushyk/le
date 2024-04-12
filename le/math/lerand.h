#ifndef __LERAND_H__
#define __LERAND_H__

#include <glib.h>

G_BEGIN_DECLS

gfloat                   le_random_uniform_f32              (void);

gfloat                   le_random_normal_f32               (void);

typedef enum LeDistribution
{
    LE_DISTRIBUTION_UNIFORM,
    LE_DISTRIBUTION_NORMAL
} LeDistribution;

gfloat                   le_random_f32                      (LeDistribution          distribution);

G_END_DECLS

#endif
