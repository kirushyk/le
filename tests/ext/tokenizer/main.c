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
  g_assert_cmpint (g_list_length (tokens), ==, 3);
  g_assert_cmpstr (le_tokenizer_decode (tokenizer, tokens), ==, "?5+");
  g_list_free (tokens);

  tokens = le_tokenizer_encode (tokenizer, "es");
  g_assert_nonnull (tokens);
  g_assert_cmpint (g_list_length (tokens), ==, 1);
  g_assert_cmpstr (le_tokenizer_decode (tokenizer, tokens), ==, "es");
  g_list_free (tokens);

  tokens = le_tokenizer_encode (tokenizer, "et");
  g_assert_nonnull (tokens);
  g_assert_cmpint (g_list_length (tokens), ==, 2);
  g_assert_cmpstr (le_tokenizer_decode (tokenizer, tokens), ==, "et");
  g_list_free (tokens);

  tokens = le_tokenizer_encode (tokenizer, "ky");
  g_assert_nonnull (tokens);
  g_assert_cmpint (g_list_length (tokens), ==, 1);
  g_assert_cmpstr (le_tokenizer_decode (tokenizer, tokens), ==, "ky");
  g_list_free (tokens);

  tokens = le_tokenizer_encode (tokenizer, "kz");
  g_assert_nonnull (tokens);
  g_assert_cmpint (g_list_length (tokens), ==, 2);
  g_assert_cmpstr (le_tokenizer_decode (tokenizer, tokens), ==, "kz");
  g_list_free (tokens);

  g_object_unref (tokenizer);

  return EXIT_SUCCESS;
}
