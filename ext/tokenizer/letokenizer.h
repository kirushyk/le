#ifndef __EXT_TOKENIZER_LETOKENIZER_H__
#define __EXT_TOKENIZER_LETOKENIZER_H__

#include <glib.h>
#include <glib-object.h>

G_BEGIN_DECLS

G_DECLARE_FINAL_TYPE (LeTokenizer, le_tokenizer, LE, TOKENIZER, GObject);

LeTokenizer * le_tokenizer_new      (const gchar * filename);

GList *       le_tokenizer_encode   (LeTokenizer * self,
                                     const gchar * text);

gchar *       le_tokenizer_decode   (LeTokenizer * self,
                                     GList *       tokens);

const gchar * le_tokenizer_decode_1 (LeTokenizer * self,
                                     guint32       token);

G_END_DECLS

#endif
