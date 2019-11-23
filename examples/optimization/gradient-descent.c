/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <math.h>
#include <le/le.h>
#include <le/optimization/legradientdescent.h>

float
J(float w)
{
    return 1 + (w - 3) * (w - 3);
}

float
dJdw(float w)
{
    return 2 * (w - 3);
}

int
main(int argc, const char *argv[])
{
    LeTensor *w = le_scalar_new_f32(0.0f);
    LeList *parameters = le_list_append(NULL, w);
    
    LeTensor *dw = le_scalar_new_f32(0.0f);
    LeList *gradients = le_list_append(NULL, dw);
    
    LeGradientDescent *optimizer = le_gradient_descent_new(parameters, 0.2f);
    ((LeOptimizer *)optimizer)->gradients = gradients;
    
    unsigned i;
    for (i = 0; i < 20; i++)
    {
        printf("Iteration %u. ", i);
        le_optimizer_step((LeOptimizer *)optimizer);
        float w_ = le_tensor_f32_at(w, 0);
        le_tensor_f32_set(dw, 0, dJdw(w_));
        float grad_ = le_tensor_f32_at(dw, 0);
        printf("J(%0.2f) = %0.2f. Gradient = %0.2f\n", w_, J(w_), grad_);
    }
    return EXIT_SUCCESS;
}