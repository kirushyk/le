#include "letokenizer.h"

struct _LeTokenizer
{
  GObject parent;
};

typedef struct _LeTokenizerPrivate
{

} LeTokenizerPrivate;

static void le_tokenizer_class_init (LeTokenizerClass *klass);
static void le_tokenizer_init (LeTokenizer *self);
G_DEFINE_FINAL_TYPE_WITH_PRIVATE (LeTokenizer, le_tokenizer, G_TYPE_OBJECT);

static void
le_tokenizer_dispose (GObject *object)
{
  LeTokenizer *self = LE_TOKENIZER (object);
  g_assert_nonnull (self);
  LeTokenizerPrivate *priv = le_tokenizer_get_instance_private (self);
  g_assert_nonnull (priv);
  G_OBJECT_CLASS (le_tokenizer_parent_class)->dispose (object);
}

static void
le_tokenizer_class_init (LeTokenizerClass *klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_tokenizer_dispose;
}

static void
le_tokenizer_init (LeTokenizer *self)
{
}

LeTokenizer *
le_tokenizer_new (const gchar *filename)
{
  LeTokenizer *self = g_object_new (le_tokenizer_get_type (), NULL);
  return self;
}
