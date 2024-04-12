/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leloss.h"
#include <le/math/leclamp.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include "tensors/letensor-imp.h"

#define EPSILON 1e-5f

gfloat
le_logistic_loss(const LeTensor *h, const LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(le_shape_equal(h->shape, y->shape));
    assert(h->shape->sizes[0] == 1);
    
    gfloat result = 0.0f;
    unsigned i;
    
    unsigned elements_count = le_shape_get_elements_count(h->shape);
    for (i = 0; i < elements_count; i++)
    {
        gfloat yi = le_tensor_at_f32(y, i);
        gfloat hi = le_clamp_f32(le_tensor_at_f32(h, i), EPSILON, 1.0f - EPSILON);
        if (yi > 0)
            result -= yi * logf(hi);
        if (yi < 1)
            result -= (1.0f - yi) * logf(1.0f - hi);
    }
    
    return result / elements_count;
}

gfloat
le_cross_entropy_loss(const LeTensor *h, const LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(le_shape_equal(h->shape, y->shape));
    assert(h->shape->sizes[0] >= 2); 
    assert(h->element_type == LE_TYPE_FLOAT32);
    assert(y->element_type == LE_TYPE_FLOAT32);
    
    unsigned num_classes = y->shape->sizes[0];
    unsigned num_examples = y->shape->sizes[1];
    
    gfloat cost = 0.0f;
    for (unsigned i = 0; i < num_examples; i++)
    {
        gfloat loss = 0.0f;
        for (unsigned j = 0; j < num_classes; j++)
        {
            gfloat y_ji = le_matrix_at_f32(y, j, i);
            gfloat h_ji = le_clamp_f32(le_matrix_at_f32(h, j, i), EPSILON, 1.0f - EPSILON);
            if (y_ji != 0.0f)
                loss -= y_ji * logf(h_ji);
        }
        cost += loss;
    }
    
    return cost / num_examples;
}

gfloat
le_mse_loss(const LeTensor *h, const LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(le_shape_equal(h->shape, y->shape));
    assert(h->element_type == LE_TYPE_FLOAT32);
    assert(y->element_type == LE_TYPE_FLOAT32);

    gfloat mse = 0.0;
    unsigned elements_count = le_shape_get_elements_count(h->shape);
    /// @todo: Speed up this
    for (unsigned i = 0; i < elements_count; i++)
    {
        gfloat d = le_tensor_at_f32(h, i) - le_tensor_at_f32(y, i);
        mse += d * d;
    }
    
    return mse / elements_count;
}

gfloat
le_one_hot_misclassification(const LeTensor *h, const LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(h->shape->sizes[0] == y->shape->sizes[0]);
    assert(h->shape->sizes[1] == y->shape->sizes[1]);
    assert(h->element_type == LE_TYPE_FLOAT32);
    assert(y->element_type == LE_TYPE_FLOAT32);
    
    unsigned i, j;
    
    unsigned classes_count = y->shape->sizes[0];
    unsigned examples_count = y->shape->sizes[1];
    unsigned misclassified_count = 0;
    
    for (i = 0; i < examples_count; i++)
    {
        int predicted_class = -2;
        gfloat predicted_class_probability = 0.0f;
        int labeled_class = -1;
        gfloat labeled_class_probability = 0.0f;
        for (j = 0; j < classes_count; j++)
        {
            gfloat predicted_probability = le_matrix_at_f32(h, j, i);
            if (predicted_probability > predicted_class_probability)
            {
                predicted_class_probability = predicted_probability;
                predicted_class = j;
            }
            gfloat labeled_probability = le_matrix_at_f32(y, j, i);
            if (labeled_probability > labeled_class_probability)
            {
                labeled_class_probability = labeled_probability;
                labeled_class = j;
            }
        }
        if (predicted_class != labeled_class)
        {
            misclassified_count++;
        }
    }
    
    return ((gfloat)misclassified_count) / ((gfloat)examples_count);
}

void
le_apply_cross_entropy_loss_derivative(LeTensor *h, const LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(le_shape_equal(h->shape, y->shape));

    unsigned i;
    
    unsigned elements_count = le_shape_get_elements_count(h->shape);
    for (i = 0; i < elements_count; i++)
    {
        gfloat yi = le_tensor_at_f32(y, i);
        gfloat hi = le_tensor_at_f32(h, i); /// @note: hi ∈ (0, 1)
        if (hi < EPSILON)
            hi = EPSILON;
        gfloat dJ_dh = (yi == 0.0f) ? 0.0f : (-yi / hi);
        le_tensor_set_f32(h, i, dJ_dh);
    }
}

void
le_apply_mse_loss_derivative(LeTensor *h, const LeTensor *y)
{
    le_tensor_sub(h, y);
}

void
le_apply_logistic_loss_derivative(LeTensor *h, const LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(le_shape_equal(h->shape, y->shape));

    unsigned i;
    
    unsigned elements_count = le_shape_get_elements_count(h->shape);
    for (i = 0; i < elements_count; i++)
    {
        gfloat yi = le_tensor_at_f32(y, i);
        gfloat hi = le_tensor_at_f32(h, i); /// @note: hi ∈ (0, 1)
        gfloat denom = hi * (1.0f - hi);
        if (denom < EPSILON)
            denom = EPSILON;
        gfloat dJ_dh = (hi == yi) ? 0 : ((hi - yi) / denom);
        le_tensor_set_f32(h, i, dJ_dh);
    }
}

gfloat 
le_loss(LeLoss loss, const LeTensor *predictions, const LeTensor *labels)
{
    switch (loss) 
    {
    case LE_LOSS_LOGISTIC:
        return le_logistic_loss(predictions, labels);
    case LE_LOSS_CROSS_ENTROPY:
        return le_cross_entropy_loss(predictions, labels);
    case LE_LOSS_MSE:
        return le_mse_loss(predictions, labels);
    default:
        return 0.0f;
    }
}

void
le_apply_loss_derivative(LeLoss loss, LeTensor *predictions, const LeTensor *labels)
{
    switch (loss) 
    {
    case LE_LOSS_LOGISTIC:
        le_apply_logistic_loss_derivative(predictions, labels);
        break;
    case LE_LOSS_CROSS_ENTROPY:
        le_apply_cross_entropy_loss_derivative(predictions, labels);
        break;
    case LE_LOSS_MSE:
        le_apply_mse_loss_derivative(predictions, labels);
        break;
    default:
        break;
    }
}

static const char * logistic_loss_name = "Logistic Loss";

static const char * xent_loss_name = "Cross-Entropy Loss";

static const char * mse_loss_name = "Mean Squared Error Loss";

const char *
le_loss_get_desc(LeLoss loss)
{
    switch (loss) 
    {
    case LE_LOSS_LOGISTIC:
        return logistic_loss_name;
    case LE_LOSS_CROSS_ENTROPY:
        return xent_loss_name;
    case LE_LOSS_MSE:
        return mse_loss_name;
    default:
        break;
    }
    return NULL;
}
