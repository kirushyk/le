#ifndef __LEMETAL_H__
#define __LEMETAL_H__

#include <le/le.h>

void le_metal_init(void);

LeTensor *
le_metal_matrix_new_product(const LeTensor *a, bool transpose_a, const LeTensor *b, bool transpose_b);

LeTensor *
le_tensor_to_metal(const LeTensor *another);

LeTensor *
le_tensor_to_cpu(const LeTensor *another);

void *
le_metal_data_copy(void *data, gsize bytes);

void
le_metal_tensor_mul_tensor(LeTensor *self, const LeTensor *b);

void le_metal_tensor_apply_sigmoid (LeTensor *tensor);

void le_metal_tensor_apply_sigmoid_prime (LeTensor *tensor);

void
le_metal_data_free(void *data);

#endif
