#include <stdlib.h>
#include <gtk/gtk.h>
#include <le/le.h>
#include "menus.h"
#include "main-window.h"

static void
quit_activated (GSimpleAction *action, GVariant *parameter, gpointer application)
{
    g_application_quit (G_APPLICATION (application));
}

static GActionEntry app_entries[] =
{
    { "quit", quit_activated, NULL }
};

static void
le_activate (GtkApplication *application, gpointer user_data)
{
    GtkWidget *window;
    
    window = le_main_window_new (application);
    
    gtk_widget_show_all (window);

}


static void
le_startup (GApplication *application, gpointer user_data)
{
    gtk_application_set_app_menu (GTK_APPLICATION (application), le_app_menu_new ());
    gtk_application_set_menubar (GTK_APPLICATION (application), le_menubar_new ());
    
    g_action_map_add_action_entries (G_ACTION_MAP (application), app_entries, G_N_ELEMENTS (app_entries), application);
}

int
main(int argc, char *argv[])
{
    GtkApplication *app;
    int status;
    
#if __APPLE__
    sranddev();
#else
    srand(time(nullptr));
#endif
    
    g_set_prgname ("le-gtk-demo");
    g_set_application_name ("Le Playground");
    
    app = gtk_application_new ("org.kirushyk.le-gtk-demo", G_APPLICATION_HANDLES_OPEN);
    g_signal_connect (app, "activate", G_CALLBACK (le_activate), NULL);
    g_signal_connect (app, "startup", G_CALLBACK (le_startup), NULL);
    status = g_application_run (G_APPLICATION (app), argc, argv);
    g_object_unref (app);
    
    return status;
}
