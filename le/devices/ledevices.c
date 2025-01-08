#include "ledevices.h"

GList *
le_devices_get_all_devices (void)
{
  GList *list = NULL;
  list = g_list_prepend (list, g_strdup ("CPU"));
  return list;
}
