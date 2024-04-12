#ifndef __LESCALAR_H__
#define __LESCALAR_H__

#include <glib.h>
#include "letensor.h"

G_BEGIN_DECLS

/// Scalar is Rank 0 Tensor with single element
typedef LeTensor LeScalar;

LeScalar *         le_scalar_new_f32                       (gfloat                   scalar);

LeScalar *         le_scalar_new_f64                       (gdouble                  scalar);

#define le_scalar_new(s) _Generic(s, \
   gfloat: le_scalar_new_f32, \
   gdouble: le_scalar_new_f64 \
)(s)

G_END_DECLS

#endif
