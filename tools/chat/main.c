#include <stdlib.h>
#include <glib.h>
#include <ext/tokenizer/letokenizer.h>
#include <readline/history.h>
#include <readline/readline.h>

int
main (int argc, char *argv[])
{
  static gchar *tokenizer_config_filename = NULL;
  static GOptionEntry option_entries[] = { // clang-format off
    { "tokenizer-config", 't', 0, G_OPTION_ARG_STRING, &tokenizer_config_filename, "Tokenizer config", "TOKENIZER.JSON" },
    { NULL }
  }; // clang-format on
  g_set_prgname ("le-chat");
  g_set_application_name ("Le Chat");
  GOptionContext *option_context = g_option_context_new (NULL);
  g_option_context_add_main_entries (option_context, option_entries, NULL);
  g_option_context_set_summary (option_context, g_get_application_name ());
  GError *error = NULL;
  if (!g_option_context_parse (option_context, &argc, &argv, &error)) {
    g_assert_nonnull (error);
    g_printerr ("%s\n", error->message);
    g_error_free (error);
    return EXIT_FAILURE;
  }

  LeTokenizer *tokenizer = le_tokenizer_new (tokenizer_config_filename);
  if (tokenizer == NULL) {
    g_printerr ("Can not initialize tokenizer.\n");
    return EXIT_FAILURE;
  }

  while (TRUE) {
    gchar *prompt = readline ("> ");
    if (prompt == NULL) {
      break;
    }
    add_history (prompt);
  }

  g_object_unref (tokenizer);

  g_option_context_free (option_context);
  return EXIT_SUCCESS;
}
