#include "lescalar.h"
#include <stdlib.h>
#include "letensor-imp.h"

LeScalar *
le_scalar_new_f32(float scalar)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = le_shape_new(0);
    self->stride = 0;
    self->owns_data = true;
    self->data = malloc(sizeof(float));
    *((float *)self->data) = scalar;
    return self;
}
