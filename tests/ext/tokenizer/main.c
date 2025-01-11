#include <stdlib.h>
#include <ext/tokenizer/letokenizer.h>

int
main ()
{
  LeTokenizer *tokenizer = le_tokenizer_new ("tokenizer.json");
  g_assert_nonnull (tokenizer);
  g_object_unref (tokenizer);
  return EXIT_SUCCESS;
}
