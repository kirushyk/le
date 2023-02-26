/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <gtk/gtk.h>
#include <le/le.h>
#include "ma-menus.h"
#include "ma-main-window.h"

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
    GtkWidget *window = le_main_window_new(application);
    gtk_window_present(GTK_WINDOW(window));
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
    g_set_prgname("le-mnist-inspect");
    g_set_application_name("MNIST Mean Classes");
    
    GtkApplication *app = gtk_application_new("org.kirushyk.le-gtk-mnist-mean-classes", G_APPLICATION_HANDLES_OPEN);
    g_signal_connect(app, "activate", G_CALLBACK(le_activate), NULL);
    g_signal_connect(app, "startup", G_CALLBACK(le_startup), NULL);
    int status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);
    
    return status;
}
