/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_BOARD_FILE_H__
#define __LE_BOARD_FILE_H__

#include <glib.h>
#include <libsoup/soup.h>

G_BEGIN_DECLS

void set_error_message (SoupMessage       *msg,
                        guint16            code);

void file_callback     (SoupServer        *server,
                        SoupMessage       *msg,
                        const char        *path,
                        GHashTable        *query,
                        SoupClientContext *context,
                        gpointer           data);

G_END_DECLS

#endif
