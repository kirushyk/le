/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <math.h>
#include <le/le.h>

/** @note: Dummy cost function from single weight argument w */
gfloat
J(gfloat w)
{
    return 1 + (w - 3) * (w - 3);
}

/** @note: Derivative of that cost function with respect to its argument w */
gfloat
dJ_dw(gfloat w)
{
    return 2 * (w - 3);
}

int
main(int argc, const char *argv[])
{
    LeTensor *w = le_scalar_new(0.0f);
    GList *parameters = g_list_append(NULL, w);
    
    LeTensor *dw = le_scalar_new(0.0f);
    GList *gradients = g_list_append(NULL, dw);
    
    LeOptimizer *optimizer = LE_OPTIMIZER(le_bgd_new_simple(parameters, gradients, 0.2f));
    
    unsigned i;
    for (i = 0; i < 20; i++)
    {
        printf("Iteration %u. ", i);
        le_optimizer_step(optimizer);
        gfloat w_ = le_tensor_at_f32(w, 0);
        le_tensor_set_f32(dw, 0, dJ_dw(w_));
        gfloat grad_ = le_tensor_at_f32(dw, 0);
        printf("J(%0.2f) = %0.2f. Gradient = %0.2f\n", w_, J(w_), grad_);
    }
    
    g_object_unref(LE_BGD(optimizer));
    
    return EXIT_SUCCESS;
}
