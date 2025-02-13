#include "lecpu.h"
#include "cpu-backend-config.h"
#include "../../le/lebackend.h"
#include "../../le/tensors/letensor-imp.h"

#ifdef HAVE_ACCELERATE_FRAMEWORK
#  include <Accelerate/Accelerate.h>
#endif
#ifdef HAVE_OPENBLAS
#  include <cblas.h>
#endif

struct _LeCpuBackend
{
  GObject parent;
};

static void
le_cpu_backend_class_init (LeCpuBackendClass *klass)
{
}

static void
le_cpu_backend_init (LeCpuBackend *self)
{
}

gfloat
le_cblas_dot_product (const LeTensor *a, const LeTensor *b)
{
#if defined(HAVE_ACCELERATE_FRAMEWORK) || defined(HAVE_OPENBLAS)
  return cblas_sdot (a->shape->sizes[0], a->data, a->stride, b->data, b->stride);
#endif
}

static void
le_cpu_backend_inteface_init (LeBackendInterface *iface)
{
  iface->dot_product = NULL;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (LeCpuBackend, le_cpu_backend, G_TYPE_INITIALLY_UNOWNED,
    G_IMPLEMENT_INTERFACE (le_backend_get_type (), le_cpu_backend_inteface_init))

LeCpuBackend *
le_cpu_backend_get_instance (void)
{
  static gsize init_guard = 0;
  static LeCpuBackend *self = NULL;

  if (g_once_init_enter (&init_guard)) {
    self = g_object_new (le_cpu_backend_get_type (), NULL);
    g_object_add_weak_pointer (G_OBJECT (self), (gpointer *)&self);
    g_once_init_leave (&init_guard, (gsize)self);
  }

  return g_object_ref_sink (self);
}
