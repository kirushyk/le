#include <stdlib.h>
#include <glib.h>

int
main (int argc, char *argv[])
{
  static GOptionEntry option_entries[] = {
    { "model", 'm', 0, G_OPTION_ARG_STRING, NULL, "Model filename", "FILENAME" },
    { NULL }
  };
  g_set_prgname ("le-chat");
  g_set_application_name ("Le Chat");
  GOptionContext *option_context = g_option_context_new (NULL);
  g_option_context_add_main_entries (option_context, option_entries, NULL);
  g_option_context_set_summary (option_context, g_get_application_name ());
  GError *error = NULL;
  if (!g_option_context_parse (option_context, &argc, &argv, &error)) {
    g_assert_nonnull (error);
    g_print ("%s\n", error->message);
    g_error_free (error);
    return EXIT_FAILURE;
  }
  g_option_context_free (option_context);
  return EXIT_SUCCESS;
}
