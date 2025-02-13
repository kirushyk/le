#include "letokenizer.h"
#include <json-glib/json-glib.h>

typedef struct MergeIndices
{
  guint32 first;
  guint32 second;
  guint32 merged;
} MergeIndices;

struct _LeTokenizer
{
  GObject parent;
  GHashTable *text_to_id;
  GHashTable *id_to_text;
  gsize num_merge_pairs;
  MergeIndices *merge_indices;
};

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
  if (self->merge_indices)
    g_free (self->merge_indices);
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

  GList *vocab_members = json_object_get_members (vocab_object);
  for (GList *iter = vocab_members; iter != NULL; iter = iter->next) {
    const gchar *key = (const gchar *)iter->data;
    gint64 id = json_object_get_int_member (vocab_object, key);
    GString *text_with_space = g_string_new (key);
    g_string_replace (text_with_space, "Ġ", " ", 0);
    g_hash_table_insert (self->text_to_id, g_strdup (text_with_space->str), GINT_TO_POINTER (id));
    /// @todo: `g_string_free_and_steal` is not available in previous versions of GLib
    g_hash_table_insert (self->id_to_text, GINT_TO_POINTER (id), g_string_free (text_with_space, FALSE));
  }
  g_list_free (vocab_members);

  self->num_merge_pairs = 0;
  JsonArray *merges_array = json_object_get_array_member (model_object, "merges");
  if (merges_array) {
    self->num_merge_pairs = json_array_get_length (merges_array);
    self->merge_indices = g_new (MergeIndices, self->num_merge_pairs);
    if (merges_array != NULL) {
      for (gsize i = 0; i < self->num_merge_pairs; i++) {
        const gchar *merge_string = json_array_get_string_element (merges_array, i);
        if (merge_string != NULL) {
          gchar **merge_item = g_strsplit (merge_string, " ", 2);
          self->merge_indices[i].first = GPOINTER_TO_INT (g_hash_table_lookup (self->text_to_id, merge_item[0]));
          self->merge_indices[i].second = GPOINTER_TO_INT (g_hash_table_lookup (self->text_to_id, merge_item[1]));
          gchar *merged_token = g_strjoinv ("", merge_item);
          self->merge_indices[i].merged = GPOINTER_TO_INT (g_hash_table_lookup (self->text_to_id, merged_token));
          g_free (merged_token);
          g_strfreev (merge_item);
        }
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
  g_assert_nonnull (self);
  if (text == NULL)
    return NULL;

  GList *tokens = NULL;

  GRegex *regex = g_regex_new ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
                               "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      0, 0, NULL);
  g_assert_nonnull (regex);
  GMatchInfo *match_info;
  g_regex_match (regex, text, 0, &match_info);
  while (g_match_info_matches (match_info)) {
    gchar *chunk = g_match_info_fetch (match_info, 0);
    g_assert_nonnull (chunk);
    gint64 id = GPOINTER_TO_INT (g_hash_table_lookup (self->text_to_id, chunk));
    if (id) {
      tokens = g_list_prepend (tokens, GINT_TO_POINTER (id));
    } else {
      GList *chunk_tokens = NULL;
      for (gsize i = 0; chunk[i] != '\0'; i++) {
        gchar *token = g_strndup (chunk + i, 1);
        gint64 id = GPOINTER_TO_INT (g_hash_table_lookup (self->text_to_id, token));
        if (id) {
          chunk_tokens = g_list_prepend (chunk_tokens, GINT_TO_POINTER (id));
        }
        g_free (token);
      }

      guint num_tokens_merged;
      do {
        num_tokens_merged = 0;
        for (GList *iter = chunk_tokens; iter != NULL; iter = iter->next) {
          GList *next_iter = iter->next;
          if (next_iter != NULL) {
            gint64 second_id = GPOINTER_TO_INT (iter->data);
            gint64 first_id = GPOINTER_TO_INT (next_iter->data);
            for (gsize i = 0; i < self->num_merge_pairs; i++) {
              if (self->merge_indices[i].first == first_id && self->merge_indices[i].second == second_id) {
                iter->data = GINT_TO_POINTER (self->merge_indices[i].merged);
                chunk_tokens = g_list_delete_link (chunk_tokens, next_iter);
                num_tokens_merged++;
                break;
              }
            }
          }
        }
      } while (num_tokens_merged > 0);

      tokens = g_list_concat (chunk_tokens, tokens);
    }
    g_free (chunk);
    g_match_info_next (match_info, NULL);
  }
  g_match_info_free (match_info);
  g_regex_unref (regex);

  tokens = g_list_reverse (tokens);

  return tokens;
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
  /// @todo: `g_string_free_and_steal` is not available in previous versions of GLib
  return g_string_free (result, FALSE);
}

const gchar *
le_tokenizer_decode_1 (LeTokenizer *self, guint32 token)
{
  const gchar *text = g_hash_table_lookup (self->id_to_text, GINT_TO_POINTER (token));
  return text;
}
