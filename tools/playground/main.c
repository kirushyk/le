/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <locale.h>
#include <stdlib.h>
#include <gtk/gtk.h>
#include <le/le.h>
#include "pg-menus.h"
#include "pg-main-window.h"

static void
quit_activated(GSimpleAction *action, GVariant *parameter, gpointer application)
{
    g_application_quit(G_APPLICATION(application));
}

static GActionEntry app_entries[] =
{
    { "quit", quit_activated, NULL }
};

static void
le_activate(GtkApplication *application, gpointer user_data)
{
    GtkWidget *window;
    window = le_main_window_new(application);
    gtk_window_present(GTK_WINDOW(window));
    le_main_window_set_preffered_model(window, PREFERRED_MODEL_TYPE_NEURAL_NETWORK);
}

static void
le_startup(GApplication *application, gpointer user_data)
{
    // gtk_application_set_app_menu(GTK_APPLICATION(application), le_app_menu_new());
    gtk_application_set_menubar(GTK_APPLICATION(application), le_menubar_new());
    
    g_action_map_add_action_entries(G_ACTION_MAP(application), app_entries, G_N_ELEMENTS(app_entries), application);
}

int
main(int argc, char *argv[])
{    
#if __APPLE__
    sranddev();
#else
    srand(time(NULL));
#endif
    setlocale(LC_NUMERIC, "C");
    
    g_set_prgname("le-gtk-demo");
    g_set_application_name("Le Playground");
    
    GtkApplication *app = gtk_application_new("com.github.kirushyk.le.gtk-demo", G_APPLICATION_HANDLES_OPEN);
    g_signal_connect(app, "activate", G_CALLBACK(le_activate), NULL);
    g_signal_connect(app, "startup", G_CALLBACK(le_startup), NULL);
    int status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);
    
    return status;
}
