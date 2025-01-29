#include "lecpu.h"

struct _LeCpuBackend
{
  GObject parent;
};

static void le_cpu_backend_class_init (LeCpuBackendClass *klass);
static void le_cpu_backend_init (LeCpuBackend *self);
G_DEFINE_FINAL_TYPE (LeCpuBackend, le_cpu_backend, G_TYPE_OBJECT)

static void
le_cpu_backend_class_init (LeCpuBackendClass *klass)
{
}

static void
le_cpu_backend_init (LeCpuBackend *self)
{
}