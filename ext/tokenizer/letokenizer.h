#ifndef __EXT_TOKENIZER_LETOKENIZER_H__
#define __EXT_TOKENIZER_LETOKENIZER_H__

#include <glib.h>
#include <glib-object.h>

G_BEGIN_DECLS

G_DECLARE_FINAL_TYPE (LeTokenizer, le_tokenizer, LE, TOKENIZER, GObject);

LeTokenizer * le_tokenizer_new (const gchar * filename);

G_END_DECLS

#endif
