#ifndef __BACKENDS_METAL_LEMETAL_H__
#define __BACKENDS_METAL_LEMETAL_H__

#include <glib.h>
#include <le/le.h>

G_BEGIN_DECLS

void       le_metal_init                       (void);

GList *    le_metal_get_all_devices            (void);

LeTensor * le_metal_matrix_new_product         (const LeTensor * a,
                                                bool             transpose_a,
                                                const LeTensor * b,
                                                bool             transpose_b);

LeTensor * le_tensor_to_metal                  (const LeTensor * another);

LeTensor * le_tensor_to_cpu                    (const LeTensor * another);

void *     le_metal_data_copy                  (void *           data,
                                                gsize            bytes);

void       le_metal_tensor_mul_tensor          (LeTensor *       self,
                                                const LeTensor * b);

void       le_metal_tensor_apply_sigmoid       (LeTensor *       tensor);

void       le_metal_tensor_apply_sigmoid_prime (LeTensor *       tensor);

void       le_metal_data_free                  (void *           data);

G_END_DECLS

#endif
