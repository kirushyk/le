#include <stdlib.h>
#include <glib.h>
#include <le/le.h>
#include <ext/tokenizer/letokenizer.h>
#include <readline/history.h>
#include <readline/readline.h>

int
main (int argc, char *argv[])
{
  static gchar *tokenizer_config_filename = NULL;
  static gboolean debug_tokenizer = FALSE;
  static GOptionEntry option_entries[] = { // clang-format off
    { "tokenizer-config", 't', 0, G_OPTION_ARG_STRING, &tokenizer_config_filename, "Tokenizer config", "TOKENIZER.JSON" },
    { "debug-tokenizer", '\0', 0, G_OPTION_ARG_NONE, &debug_tokenizer, "Whether to debug tokenization", NULL },
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
  
  LeTensor *tokens_embeddings = le_matrix_new_rand_f32 (LE_DISTRIBUTION_UNIFORM, 2048, 128256);

  while (TRUE) {
    gchar *prompt = readline ("> ");
    if (prompt == NULL) {
      break;
    }
    add_history (prompt);
    GList *tokens = le_tokenizer_encode (tokenizer, prompt);
    if (debug_tokenizer) {
      g_print ("t ");
      gint prev_color = -1;
      gint prev_underline = 0;
      for (GList *iter = tokens; iter != NULL; iter = iter->next) {
        const guint32 id = GPOINTER_TO_INT (iter->data);
        gint color = 41 + (id % 6);
        gint underline = prev_underline ? 0 : (color == prev_color ? 4 : 0);
        g_print ("\033[%d;%dm%s\033[0m", underline, color, le_tokenizer_decode_1 (tokenizer, id));
        prev_color = color;
        prev_underline = underline;
      }
      g_print ("\n");
    }
    gchar *prompt_decoded = le_tokenizer_decode (tokenizer, tokens);
    g_print ("> %s\n", prompt_decoded);
    g_list_free (tokens);
  }

  le_tensor_unref (tokens_embeddings);
  g_object_unref (tokenizer);

  g_option_context_free (option_context);
  return EXIT_SUCCESS;
}
