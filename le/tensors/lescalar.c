#include "lescalar.h"
#include <stdlib.h>
#include "letensor-imp.h"

LeScalar *
le_scalar_new_f32(float scalar)
{
    LeTensor *self = g_new0 (LeTensor, 1);
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = le_shape_new(0);
    self->stride = 0;
    self->owns_data = true;
    self->data = g_new0 (gfloat, 1);
    *((float *)self->data) = scalar;
    return self;
}

LeScalar *
le_scalar_new_f64(double scalar)
{
    LeTensor *self = g_new0 (LeTensor, 1);
    self->element_type = LE_TYPE_FLOAT64;
    self->shape = le_shape_new(0);
    self->stride = 0;
    self->owns_data = true;
    self->data = g_new0 (gdouble, 1);
    *((double *)self->data) = scalar;
    return self;
}
