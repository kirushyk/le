#include "ledevices.h"
#include "../config.h"
#include <le/le.h>
#ifdef HAVE_METAL
#  include "../backends/metal/lemetal.h"
#endif
#ifdef HAVE_CUDA
#  include "../backends/cuda/lecuda.h"
#endif

GList *
le_devices_get_all_devices (void)
{
  GList *list = NULL;
#ifdef HAVE_METAL
  list = g_list_concat (list, le_metal_get_all_devices ());
#endif
#ifdef HAVE_CUDA
  list = g_list_concat (list, le_cuda_get_all_devices ());
#endif
  return list;
}
