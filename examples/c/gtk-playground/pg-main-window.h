/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __GTK_PLAYGROUND_PG_MAIN_WINDOW_H__
#define __GTK_PLAYGROUND_PG_MAIN_WINDOW_H__
#include <gtk/gtk.h>

typedef enum PreferredModelType
{
  PREFERRED_MODEL_TYPE_POLYNOMIAL_REGRESSION,
  PREFERRED_MODEL_TYPE_SUPPORT_VECTOR_MACHINE,
  PREFERRED_MODEL_TYPE_NEURAL_NETWORK,
  PREFERRED_MODEL_TYPE_KNN
} PreferredModelType;

GtkWidget * le_main_window_new                 (GtkApplication *   app);

void        le_main_window_set_preffered_model (GtkWidget *        self,
                                                PreferredModelType model_type);

#endif
