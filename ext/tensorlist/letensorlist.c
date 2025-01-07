/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "tensorlist"

#include "letensorlist.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <glib.h>
#include <le/tensors/letensor-imp.h>

static void
le_tensor_serialize (LeTensor *tensor, FILE *fout)
{
  g_assert_cmpint (tensor->device_type, ==, LE_DEVICE_TYPE_CPU);
  g_assert_nonnull (tensor);
  g_assert_nonnull (fout);

  fwrite ((guint8 *)&tensor->element_type, sizeof (guint8), 1, fout);
  fwrite ((guint8 *)&tensor->shape->num_dimensions, sizeof (guint8), 1, fout);
  fwrite (tensor->shape->sizes, sizeof (guint32), tensor->shape->num_dimensions, fout);
  gsize elements_count = le_shape_get_elements_count (tensor->shape);
  fwrite (tensor->data, le_type_size (tensor->element_type), elements_count, fout);
}

void
le_tensorlist_save (GList *tensors, const char *filename)
{
  FILE *fout = fopen (filename, "wb");
  if (fout) {
    guint8 version = 2;
    fwrite (&version, sizeof (version), 1, fout);
    guint16 num_tensors = 0;
    /// @todo: Make this be GList function
    for (GList *current = tensors; current != NULL; current = current->next)
      num_tensors++;
    fwrite (&num_tensors, sizeof (num_tensors), 1, fout);
    for (GList *current = tensors; current != NULL; current = current->next) {
      le_tensor_serialize (LE_TENSOR (current->data), fout);
    }
    fclose (fout);
  }
}

static LeTensor *
le_tensor_deserialize (FILE *fin)
{
  g_assert_nonnull (fin);

  LeTensor *self = g_new0 (LeTensor, 1);
  fread ((guint8 *)&self->element_type, sizeof (guint8), 1, fin);

  self->shape = g_new0 (LeShape, 1);
  fread ((guint8 *)&self->shape->num_dimensions, sizeof (guint8), 1, fin);
  self->shape->sizes = g_new0 (gsize, self->shape->num_dimensions);
  fread (self->shape->sizes, sizeof (guint32), self->shape->num_dimensions, fin);

  if (self->shape->num_dimensions > 0)
    self->stride = le_shape_get_size (self->shape, -1);
  else
    self->stride = 0;
  self->owns_data      = true;
  self->device_type    = LE_DEVICE_TYPE_CPU;
  gsize elements_count = le_shape_get_elements_count (self->shape);
  self->data           = g_malloc (elements_count * le_type_size (self->element_type));
  fread (self->data, le_type_size (self->element_type), elements_count, fin);

  return self;
}

GList *
le_tensorlist_load (const char *filename)
{
  GList *list = NULL;
  FILE  *fin  = fopen (filename, "rb");
  if (fin) {
    guint8 version = 0;
    fread (&version, sizeof (version), 1, fin);
    if (version <= 2) {
      guint16 num_tensors = 0;
      fread (&num_tensors, sizeof (num_tensors), 1, fin);
      for (guint16 i = 0; i < num_tensors; i++) {
        LeTensor *tensor = le_tensor_deserialize (fin);
        list             = g_list_append (list, tensor);
      }
    } else {
      LE_WARNING ("%s: Unknown version of .tensorlist file: %d", filename, (int)version);
    }
    fclose (fin);
  } else {
    LE_WARNING ("File not found: %s", filename);
  }
  return list;
}
