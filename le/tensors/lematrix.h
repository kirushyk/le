/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

/* Linear Algebra for two-dimensional Tensors */

#ifndef __LEMATRIX_H__
#define __LEMATRIX_H__

#include <le/lemacros.h>
#include <le/math/lerand.h>
#include "letensor.h"

LE_BEGIN_DECLS

unsigned           le_matrix_get_width                     (const LeTensor *        matrix);

unsigned           le_matrix_get_height                    (const LeTensor *        matrix);

float              le_matrix_at_f32                        (const LeTensor *        matrix,
                                                            unsigned                y,
                                                            unsigned                x);

double             le_matrix_at_f64                        (const LeTensor *        matrix,
                                                            unsigned                y,
                                                            unsigned                x);   

int8_t             le_matrix_at_i8                         (const LeTensor *        matrix,
                                                            unsigned                y,
                                                            unsigned                x);

int16_t            le_matrix_at_i16                        (const LeTensor *        matrix,
                                                            unsigned                y,
                                                            unsigned                x);

int32_t            le_matrix_at_i32                        (const LeTensor *        matrix,
                                                            unsigned                y,
                                                            unsigned                x);

uint32_t           le_matrix_at_u32                        (const LeTensor *        matrix,
                                                            unsigned                y,
                                                            unsigned                x);                             

/// @note: Unlike le_tensor_add, supports horizontal broadcasting
void               le_matrix_add                           (LeTensor *              matrix,
                                                            const LeTensor *        another);

void               le_matrix_set_i8                        (LeTensor *              matrix,
                                                            unsigned                y,
                                                            unsigned                x,
                                                            int8_t                  value);

void               le_matrix_set_u8                        (LeTensor *              matrix,
                                                            unsigned                y,
                                                            unsigned                x,
                                                            uint8_t                 value);

void               le_matrix_set_i16                       (LeTensor *              matrix,
                                                            unsigned                y,
                                                            unsigned                x,
                                                            int16_t                 value);

void               le_matrix_set_u16                       (LeTensor *              matrix,
                                                            unsigned                y,
                                                            unsigned                x,
                                                            uint16_t                value);

void               le_matrix_set_i32                       (LeTensor *              matrix,
                                                            unsigned                y,
                                                            unsigned                x,
                                                            int32_t                 value);

void               le_matrix_set_u32                       (LeTensor *              matrix,
                                                            unsigned                y,
                                                            unsigned                x,
                                                            uint32_t                value);

void               le_matrix_set_f16                       (LeTensor *              matrix,
                                                            unsigned                y,
                                                            unsigned                x,
                                                            lehalf                  value);

void               le_matrix_set_f32                       (LeTensor *              matrix,
                                                            unsigned                y,
                                                            unsigned                x,
                                                            float                   value);

void               le_matrix_set_f64                       (LeTensor *              matrix,
                                                            unsigned                y,
                                                            unsigned                x,
                                                            double                  value);

/// @note: Half is not accepted here
/// @note: Make sure to pass correct argument type here
#define le_matrix_set(m, y, x, v) _Generic((v), \
   int8_t: le_matrix_set_i8, \
   uint8_t: le_matrix_set_u8, \
   int16_t: le_matrix_set_i16, \
   uint16_t: le_matrix_set_u16, \
   int32_t: le_matrix_set_i32, \
   uint32_t: le_matrix_set_u32, \
   float: le_matrix_set_f32, \
   double: le_matrix_set_f64 \
)(m, y, x, v)

LeTensor *         le_matrix_new_identity                  (LeType                  type,
                                                            unsigned                size);

LeTensor *         le_matrix_new_uninitialized             (LeType                  type,
                                                            unsigned                height,
                                                            unsigned                width);

LeTensor *         le_matrix_new_zeros                     (LeType                  type,
                                                            unsigned                height,
                                                            unsigned                width);

LeTensor *         le_matrix_new_rand_f32                  (LeDistribution          distribution,
                                                            unsigned                height,
                                                            unsigned                width);

LeTensor *         le_matrix_new_transpose                 (LeTensor *              a);

LeTensor *         le_matrix_new_sum                       (const LeTensor *        a,
                                                            unsigned                dimension);

LeTensor *         le_matrix_new_one_hot                   (LeType                  type,
                                                            const LeTensor *        a,
                                                            unsigned                num_classes);

LeTensor *         le_matrix_new_product                   (const LeTensor *        a,
                                                            const LeTensor *        b);

LeTensor *         le_matrix_new_product_full              (const LeTensor *        a,
                                                            bool                    transpose_a,
                                                            const LeTensor *        b,
                                                            bool                    transpose_b);

                                            
LeTensor *         le_matrix_new_conv2d                    (const LeTensor *        image,
                                                            const LeTensor *        filter);

LeTensor *         le_matrix_get_column                    (const LeTensor *        matrix,
                                                            unsigned                x);

LeTensor *         le_matrix_get_column_copy               (const LeTensor *        matrix,
                                                            unsigned                x);

LeTensor *         le_matrix_get_columns_copy              (const LeTensor *        matrix,
                                                            unsigned                x,
                                                            unsigned                width);

void               le_matrix_apply_softmax                 (LeTensor *              matrix);

LE_END_DECLS

#endif
