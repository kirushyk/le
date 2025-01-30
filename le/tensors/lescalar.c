#include "lescalar.h"
#include <stdlib.h>
#include "letensor-imp.h"

LeScalar *
le_scalar_new_f32(gfloat scalar)
{
    LeTensor *self = g_new0 (LeTensor, 1);
    self->element_type = LE_TYPE_F32;
    self->shape = le_shape_new(0);
    self->stride = 0;
    self->owns_data = true;
    self->data = g_new0 (gfloat, 1);
    *((gfloat *)self->data) = scalar;
    return self;
}

LeScalar *
le_scalar_new_f64(gdouble scalar)
{
    LeTensor *self = g_new0 (LeTensor, 1);
    self->element_type = LE_TYPE_F64;
    self->shape = le_shape_new(0);
    self->stride = 0;
    self->owns_data = true;
    self->data = g_new0 (gdouble, 1);
    *((gdouble *)self->data) = scalar;
    return self;
}
