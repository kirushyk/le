/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leloss.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include "letensor-imp.h"

#define EPSILON 1e-5f

float
le_logistic_loss(const LeTensor *h, const LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(le_shape_equal(h->shape, y->shape));
    assert(h->shape->sizes[0] == 1);
    
    float result = 0.0f;
    unsigned i;
    
    unsigned elements_count = le_shape_get_elements_count(h->shape);
    for (i = 0; i < elements_count; i++)
    {
        float yi = le_tensor_at_f32(y, i);
        float hi = le_tensor_at_f32(h, i);
        result -= yi * logf(hi) + (1.0f - yi) * logf(1.0f - hi);
    }
    
    return result / elements_count;
}

float
le_cross_entropy_loss(const LeTensor *h, const LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(le_shape_equal(h->shape, y->shape));
    assert(h->shape->sizes[0] >= 2);
    
    unsigned num_classes = y->shape->sizes[0];
    unsigned num_examples = y->shape->sizes[1];
    
    float cost = 0.0f;
    for (unsigned i = 0; i < num_examples; i++)
    {
        float loss = 0.0f;
        for (unsigned j = 0; j < num_classes; j++)
        {
            float y_ji = le_matrix_at_f32(y, j, i);
            float h_ji = le_matrix_at_f32(h, j, i);
            loss -= y_ji * logf(h_ji);
        }
        cost += loss;
    }
    
    return cost / num_examples;
}

float
le_one_hot_misclassification(const LeTensor *h, const LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(h->shape->sizes[1] == y->shape->sizes[1]);
    
    unsigned i, j;
    
    unsigned classes_count = y->shape->sizes[0];
    unsigned examples_count = y->shape->sizes[1];
    unsigned misclassified_count = 0;
    
    for (i = 0; i < examples_count; i++)
    {
        int predicted_class = -2;
        float predicted_class_probability = 0.0f;
        int labeled_class = -1;
        float labeled_class_probability = 0.0f;
        for (j = 0; j < classes_count; j++)
        {
            float predicted_probability = le_matrix_at_f32(h, j, i);
            if (predicted_probability > predicted_class_probability)
            {
                predicted_class_probability = predicted_probability;
                predicted_class = j;
            }
            float labeled_probability = le_matrix_at_f32(y, j, i);
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
    
    return ((float)misclassified_count) / ((float)examples_count);
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
        float yi = le_tensor_at_f32(y, i * y->stride);
        float hi = le_tensor_at_f32(h, i * h->stride); /// @note: hi ∈ (0, 1)
        if (hi < EPSILON)
            hi = EPSILON;
        float dJ_dh = (yi == 0) ? 0 : (-yi / hi);
        le_tensor_set_f32(h, i * h->stride, dJ_dh);
    }
}

void
le_apply_mse_loss_derivative(LeTensor *h, const LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(le_shape_equal(h->shape, y->shape));

    unsigned i;
    
    unsigned elements_count = le_shape_get_elements_count(h->shape);
    for (i = 0; i < elements_count; i++)
    {
        float yi = le_tensor_at_f32(y, i);
        float hi = le_tensor_at_f32(h, i);
        float dJ_dh = hi - yi;
        le_tensor_set_f32(h, i, dJ_dh);
    }
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
        float yi = le_tensor_at_f32(y, i);
        float hi = le_tensor_at_f32(h, i); /// @note: hi ∈ (0, 1)
        float denom = hi * (1.0f - hi);
        if (denom < EPSILON)
            denom = EPSILON;
        float dJ_dh = (hi == yi) ? 0 : ((hi - yi) / denom);
        le_tensor_set_f32(h, i, dJ_dh);
    }
}

float 
le_loss(LeLoss loss, const LeTensor *predictions, const LeTensor *labels)
{
    switch (loss) 
    {
    case LE_LOSS_LOGISTIC:
        return le_logistic_loss(predictions, labels);
    case LE_LOSS_CROSS_ENTROPY:
        return le_cross_entropy_loss(predictions, labels);
    case LE_LOSS_MSE:
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
