/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef _LE_MAIN_WINDOW_
#   define _LE_MAIN_WINDOW_
#   include <gtk/gtk.h>

typedef enum PreferredModelType
{
    PREFERRED_MODEL_TYPE_POLYNOMIAL_REGRESSION,
    PREFERRED_MODEL_TYPE_SUPPORT_VECTOR_MACHINE,
    PREFERRED_MODEL_TYPE_NEURAL_NETWORK
} PreferredModelType;

GtkWidget * le_main_window_new (GtkApplication *app);

void le_main_window_set_preffered_model(GtkWidget *self, PreferredModelType model_type);

#endif
