#include <stdlib.h>
#include <glib.h>
#include <ext/tokenizer/letokenizer.h>

int
main ()
{
  GList *tokens = NULL;
  tokens = g_list_prepend (tokens, GINT_TO_POINTER (10));
  tokens = g_list_prepend (tokens, GINT_TO_POINTER (20));
  tokens = g_list_prepend (tokens, GINT_TO_POINTER (30));
  LeTokenizer *tokenizer = le_tokenizer_new ("tokenizer.json");
  g_assert_nonnull (tokenizer);
  gchar *string = le_tokenizer_decode (tokenizer, tokens);
  g_print ("%s\n", string);
  g_free (string);
  g_list_free (tokens);
  g_object_unref (tokenizer);
  return EXIT_SUCCESS;
}
