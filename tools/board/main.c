/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <glib.h>
#include <libsoup/soup.h>
#include "file.h"

int main(int argc, char *argv[])
{
    GError *error = NULL;
    GMainLoop *main_loop = g_main_loop_new(NULL, FALSE);
    g_set_prgname("le-board");
    g_set_application_name("Le Board");
    SoupServer *server = soup_server_new(SOUP_SERVER_SERVER_HEADER, "le-board ", NULL);
    soup_server_add_handler(server, "/ui", file_callback, NULL, NULL);
    const int port = 6006;
    if (soup_server_listen_all(server, port, 0, &error))
    {
        g_print("http://localhost:%d/ui\n", port);
        g_main_loop_run(main_loop);
        soup_server_disconnect(server);
    }
    else
    {
        if (error->message)
        {
            g_printerr("%s\n", error->message);
        }
        else
        {
            g_printerr("port %d could not be bound\n", port);
        }
    }
    g_object_unref(server);
    g_main_loop_unref(main_loop);
    
    return EXIT_SUCCESS;
}
