#include <stdlib.h>
#include <glib.h>
#include <ext/tokenizer/letokenizer.h>

int
main ()
{
  LeTokenizer *tokenizer = le_tokenizer_new ("tokenizer.json");

  GList *tokens = NULL;
  tokens = g_list_prepend (tokens, GINT_TO_POINTER (10));
  tokens = g_list_prepend (tokens, GINT_TO_POINTER (20));
  tokens = g_list_prepend (tokens, GINT_TO_POINTER (30));
  g_assert_nonnull (tokenizer);
  gchar *string = le_tokenizer_decode (tokenizer, tokens);
  g_assert_cmpstr (string, ==, "?5+");
  g_free (string);
  g_list_free (tokens);
  tokens = NULL;

  tokens = le_tokenizer_encode (tokenizer, NULL);
  g_assert_null (tokens);

  tokens = le_tokenizer_encode (tokenizer, "?5+");
  g_assert_nonnull (tokens);
  g_assert_cmpstr (le_tokenizer_decode (tokenizer, tokens), ==, "?5+");
  g_list_free (tokens);

  g_object_unref (tokenizer);

  return EXIT_SUCCESS;
}
