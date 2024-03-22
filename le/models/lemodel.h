/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_MODEL_H__
#define __LE_MODEL_H__

#include <glib.h>
#include <glib-object.h>
#include <le/tensors/letensor.h>

G_BEGIN_DECLS

#define LE_TYPE_MODEL (le_model_get_type ())
G_DECLARE_DERIVABLE_TYPE (LeModel, le_model, LE, MODEL, GObject);

struct _LeModelClass
{
    GObjectClass parent;
    LeTensor * (*predict)         (struct _LeModel *model, const LeTensor *x);
    GList *    (*get_gradients)   (struct _LeModel *model, const LeTensor *x, const LeTensor *y);
    float      (*train_iteration) (struct _LeModel *model);
    gpointer padding[12];
};

// #define LE_MODEL(obj) ((LeModel *)(obj))

#define LE_MODEL_CLASS(klass) ((LeModelClass *)(klass))
#define LE_MODEL_GET_CLASS(obj) (LE_MODEL_CLASS(G_OBJECT_GET_CLASS(obj)))

/** @note: This function is to be used by instances of subclasses of LeModel
    to list its trainable parameters */
void                    le_model_append_parameter          (LeModel *               model,
                                                            LeTensor *              parameter);

LeTensor *              le_model_predict                   (LeModel *               model,
                                                            const LeTensor *        x);

GList *                 le_model_get_gradients             (LeModel *               model,
                                                            const LeTensor *        x,
                                                            const LeTensor *        y);

float                   le_model_train_iteration           (LeModel *               model);

GList *                 le_model_get_parameters            (LeModel *               model);

void                    le_model_free                      (LeModel *               model);

G_END_DECLS

#endif
