#ifndef __LE__MATH__CLAMP_H__
#define __LE__MATH__CLAMP_H__

#include <glib.h>

G_BEGIN_DECLS

gfloat le_clamp_f32 (gfloat v,
                    gfloat min,
                    gfloat max);

G_END_DECLS

#endif
