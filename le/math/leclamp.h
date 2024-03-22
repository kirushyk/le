#ifndef __LE__MATH__CLAMP_H__
#define __LE__MATH__CLAMP_H__

#include <glib.h>

G_BEGIN_DECLS

float le_clamp_f32 (float v,
                    float min,
                    float max);

G_END_DECLS

#endif
