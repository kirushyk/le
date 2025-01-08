#include <stdlib.h>
#include <glib.h>
#include <le/devices/ledevices.h>

int
main (int argc, char *argv[])
{
  GList *devices = le_devices_get_all_devices ();
  for (GList *i = devices; i != NULL; i = i->next) {
    g_print ("%s\n", (gchar *)i->data);
  }
  g_list_free_full (devices, g_free);
  return EXIT_SUCCESS;
}
