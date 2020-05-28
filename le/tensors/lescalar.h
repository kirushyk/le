#ifndef __LESCALAR_H__
#define __LESCALAR_H__

#include <le/lemacros.h>
#include "letensor.h"

LE_BEGIN_DECLS

/// Scalar is Rank 0 Tensor with single element
typedef LeTensor LeScalar;

LeScalar *         le_scalar_new_f32                       (float                   scalar);

LE_END_DECLS

#endif
