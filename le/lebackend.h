#ifndef __LE_LEBACKEND_H__
#define __LE_LEBACKEND_H__

#include <glib.h>
#include <glib-object.h>

G_BEGIN_DECLS

typedef struct _LeTensor LeTensor;

G_DECLARE_INTERFACE (LeBackend, le_backend, LE, BACKEND, GObject);

struct _LeBackendInterface
{
  GTypeInterface parent_class;
  
  gfloat (*dot_product) (LeTensor *a, LeTensor *b);
};

G_END_DECLS

#endif
