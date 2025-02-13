/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __BACKENDS_CUDA_LECUDA_H__
#define __BACKENDS_CUDA_LECUDA_H__

#include <le/le.h>
#include <glib.h>

G_BEGIN_DECLS

GList *    le_cuda_get_all_devices            (void);

LeTensor * le_cuda_matrix_new_product         (const LeTensor * a,
                                               gboolean         transpose_a,
                                               const LeTensor * b,
                                               gboolean         transpose_b);

void       le_cuda_tensor_mul_tensor          (LeTensor *       self,
                                               const LeTensor * b);

void       le_cuda_tensor_apply_sigmoid       (LeTensor *       tensor);

void       le_cuda_tensor_apply_sigmoid_prime (LeTensor *       tensor);

LeTensor * le_tensor_to_cuda                  (const LeTensor * cpu_tensor);
   
LeTensor * le_cuda_tensor_to_cpu              (const LeTensor * cuda_tensor);

void *     le_cuda_data_copy                  (void *           data,
                                               gsize            bytes);

void       le_cuda_data_free                  (void *           data);

G_END_DECLS

#endif
