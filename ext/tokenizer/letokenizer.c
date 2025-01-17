#include "letokenizer.h"
#include <json-glib/json-glib.h>

struct _LeTokenizer
{
  GObject parent;
  GHashTable *text_to_id;
  GHashTable *id_to_text;
  GList *merges;
};

typedef struct MergePair
{
  gchar *first;
  gchar *second;
} MergePair;

void
merge_pair_free (MergePair *pair)
{
  if (pair) {
    g_free (pair->first);
    g_free (pair->second);
  }
  g_free (pair);
}

static void le_tokenizer_class_init (LeTokenizerClass *klass);
static void le_tokenizer_init (LeTokenizer *self);
G_DEFINE_FINAL_TYPE (LeTokenizer, le_tokenizer, G_TYPE_OBJECT);

static void
le_tokenizer_dispose (GObject *object)
{
  LeTokenizer *self = LE_TOKENIZER (object);
  g_assert_nonnull (self);
  g_assert_nonnull (self->text_to_id);
  g_hash_table_unref (self->text_to_id);
  g_assert_nonnull (self->id_to_text);
  g_hash_table_unref (self->id_to_text);
  g_list_free_full (self->merges, (GDestroyNotify)merge_pair_free);
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
  LeTokenizer *self = NULL;

  JsonParser *parser = json_parser_new_immutable ();
  GError *error = NULL;
  if (!json_parser_load_from_file (parser, filename, &error)) {
    if (error != NULL) {
      g_printerr ("%s\n", error->message);
      g_error_free (error);
      error = NULL;
    }
    goto error;
  }
  JsonNode *root_node = json_parser_get_root (parser);
  if (root_node == NULL)
    goto error;
  JsonObject *root_object = json_node_get_object (root_node);
  if (root_object == NULL)
    goto error;
  JsonObject *model_object = json_object_get_object_member (root_object, "model");
  if (model_object == NULL)
    goto error;
  JsonObject *vocab_object = json_object_get_object_member (model_object, "vocab");
  if (vocab_object == NULL)
    goto error;

  self = g_object_new (le_tokenizer_get_type (), NULL);
  self->text_to_id = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);
  self->id_to_text = g_hash_table_new_full (g_direct_hash, g_direct_equal, NULL, g_free);

  GList *vocam_members = json_object_get_members (vocab_object);
  for (GList *iter = vocam_members; iter != NULL; iter = iter->next) {
    const gchar *key = (const gchar *)iter->data;
    gint64 id = json_object_get_int_member (vocab_object, key);
    g_hash_table_insert (self->text_to_id, g_strdup (key), GINT_TO_POINTER (id));
    g_hash_table_insert (self->id_to_text, GINT_TO_POINTER (id), g_strdup (key));
  }
  g_list_free (vocam_members);

  self->merges = NULL;
  JsonArray *merges_array = json_object_get_array_member (model_object, "merges");
  if (merges_array != NULL) {
    for (gsize i = 0; i < json_array_get_length (merges_array); i++) {
      const gchar *merge_string = json_array_get_string_element (merges_array, i);
      if (merge_string != NULL) {
        gchar **merge_item = g_strsplit (merge_string, " ", 2);
        MergePair *pair = g_new (MergePair, 1);
        pair->first = g_strdup (merge_item[0]);
        pair->second = g_strdup (merge_item[1]);
        g_strfreev (merge_item);
        self->merges = g_list_prepend (self->merges, pair);
      }
    }
  }

error:
  g_object_unref (parser);
  return self;
}

GList *
le_tokenizer_encode (LeTokenizer *self, const gchar *text)
{
  GList *tokens = NULL;
  for (gsize i = 0; text != NULL && text[i] != '\0'; i++) {
    gchar *token = g_strndup (text + i, 1);
    gint64 id = GPOINTER_TO_INT (g_hash_table_lookup (self->text_to_id, token));
    if (id) {
      tokens = g_list_prepend (tokens, GINT_TO_POINTER (id));
    }
    g_free (token);
  }
  return g_list_reverse (tokens);
}

gchar *
le_tokenizer_decode (LeTokenizer *self, GList *tokens)
{
  GString *result = g_string_new ("");
  for (GList *iter = tokens; iter != NULL; iter = iter->next) {
    const gint64 id = GPOINTER_TO_INT (iter->data);
    const gchar *token = g_hash_table_lookup (self->id_to_text, GINT_TO_POINTER (id));
    if (token) {
      g_string_append (result, token);
    }
  }
  return g_string_free_and_steal (result);
}
