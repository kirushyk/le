/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LELOSS_H__
#define __LELOSS_H__

#include <le/tensors/lematrix.h>
#include "glib.h"

G_BEGIN_DECLS

gfloat le_logistic_loss                       (const LeTensor *predictions,
                                              const LeTensor *labels);

gfloat le_cross_entropy_loss                  (const LeTensor *predictions,
                                              const LeTensor *label);

gfloat le_one_hot_misclassification           (const LeTensor *predictions,
                                              const LeTensor *labels);

/// @note: Partial derivative with respect to predictions
/// predictions <- ∂J(predictions, labels) / ∂predictions
void  le_apply_cross_entropy_loss_derivative (LeTensor       *predictions,
                                              const LeTensor *labels);

/// @note: Partial derivative with respect to predictions
/// predictions <- ∂J(predictions, labels) / ∂predictions
void  le_apply_mse_loss_derivative           (LeTensor       *predictions,
                                              const LeTensor *labels);

/// @note: Partial derivative with respect to predictions
/// predictions <- ∂J(predictions, labels) / ∂predictions
void  le_apply_logistic_loss_derivative      (LeTensor       *predictions,
                                              const LeTensor *labels);

typedef enum LeLoss {
   LE_LOSS_MSE,
   LE_LOSS_LOGISTIC,
   LE_LOSS_CROSS_ENTROPY
} LeLoss;

gfloat        le_loss                  (LeLoss          loss,
                                       const LeTensor *predictions,
                                       const LeTensor *labels);

void         le_apply_loss_derivative (LeLoss          loss,
                                       LeTensor       *predictions,
                                       const LeTensor *labels);

const char * le_loss_get_desc         (LeLoss          loss);    

G_END_DECLS

#endif
