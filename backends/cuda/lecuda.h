/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <le/le.h>

#ifndef __BACKENDS_CUDA_LECUDA_H__
#define __BACKENDS_CUDA_LECUDA_H__

LeTensor * le_cuda_matrix_new_product         (const LeTensor * a,
                                               bool             transpose_a,
                                               const LeTensor * b,
                                               bool             transpose_b);

void       le_cuda_tensor_mul_tensor          (LeTensor *       self,
                                               const LeTensor * b);

void       le_cuda_tensor_apply_sigmoid       (LeTensor *       tensor);

void       le_cuda_tensor_apply_sigmoid_prime (LeTensor *       tensor);

LeTensor * le_tensor_to_cuda                  (const LeTensor * cpu_tensor);
   
LeTensor * le_cuda_tensor_to_cpu              (const LeTensor * cuda_tensor);

void *     le_cuda_data_copy                  (void *           data,
                                               gsize            bytes);

void       le_cuda_data_free                  (void *           data);

#endif
