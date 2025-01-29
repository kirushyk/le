#include "lebackend.h"

static void
le_backend_default_init (LeBackendInterface *iface)
{
  iface->dot_product = NULL;
}

G_DEFINE_INTERFACE (LeBackend, le_backend, G_TYPE_INITIALLY_UNOWNED)
