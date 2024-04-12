/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lemodel.h"
#include <assert.h>
#include <stdlib.h>
#include "lelog.h"

#define DEFAULT_LOG_CATEGORY "model"

typedef struct _LeModelPrivate
{
  GList *parameters;
} LeModelPrivate;

static void le_model_class_init (LeModelClass * klass);
static void le_model_init (LeModel * self);
G_DEFINE_TYPE_WITH_PRIVATE (LeModel, le_model, G_TYPE_OBJECT);

static void
le_model_dispose (GObject * object)
{
  G_OBJECT_CLASS (le_model_parent_class)->dispose (object);
}

static void
le_model_finalize (GObject * object)
{
   
}

static void
le_model_class_init (LeModelClass * klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_model_dispose;
  G_OBJECT_CLASS (klass)->finalize = le_model_finalize;
}

static void
le_model_init (LeModel * self)
{
    LeModelPrivate *priv = le_model_get_instance_private (self);
    priv->parameters = NULL;
}

void
le_model_append_parameter(LeModel *self, LeTensor *parameter)
{
    LeModelPrivate *priv = le_model_get_instance_private (self);
    priv->parameters = g_list_append(priv->parameters, parameter);
}

LeTensor *
le_model_predict(LeModel *self, const LeTensor *x)
{
    assert(self);
    assert(G_OBJECT_GET_CLASS(self));
    assert(LE_MODEL_GET_CLASS(self)->predict);
    
    return LE_MODEL_GET_CLASS(self)->predict(self, x);
}

GList *
le_model_get_gradients(LeModel *self, const LeTensor *x, const LeTensor *y)
{
    assert(self);
    assert(G_OBJECT_GET_CLASS(self));
    
    if (LE_MODEL_GET_CLASS(self)->get_gradients == NULL)
    {
        LE_WARNING("`get_gradients` virtual function is not set in subclass");
        return NULL;
    };
    
    return LE_MODEL_GET_CLASS(self)->get_gradients(self, x, y);
}

gfloat
le_model_train_iteration(LeModel *self)
{
    return LE_MODEL_GET_CLASS(self)->train_iteration(self);
}

GList *
le_model_get_parameters(LeModel *self)
{
    LeModelPrivate *priv = le_model_get_instance_private (self);
    return priv->parameters;
}

// void
// le_model_free(LeModel *self)
// {
//     g_free (self);
// }
