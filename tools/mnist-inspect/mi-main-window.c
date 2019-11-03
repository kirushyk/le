/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "pg-main-window.h"
#include <stdlib.h>
#include <le/le.h>
#include <math.h>
#include "pg-generate-data.h"
#include "pg-color.h"

#define LE_TYPE_MAIN_WINDOW le_main_window_get_type()
G_DECLARE_FINAL_TYPE(LEMainWindow, le_main_window, LE, MAIN_WINDOW, GtkApplicationWindow);

struct _LEMainWindow
{
    GtkApplicationWindow parent_instance;
    
    GtkWidget *drawing_area;
    GtkWidget *set_selection_combo;
    GtkWidget *index_spin_button;
    GtkWidget *label_entry;

};

G_DEFINE_TYPE(LEMainWindow, le_main_window, GTK_TYPE_APPLICATION_WINDOW);

static void
close_activated(GSimpleAction *action, GVariant *parameter, gpointer data)
{
    GtkWidget *window = GTK_WIDGET(data);
    g_object_set(window, "application", NULL, NULL);
    gtk_window_close(GTK_WINDOW(window));
}

static GActionEntry win_entries[] =
{
    { "close", close_activated }
};

static void
le_main_window_constructed(GObject *object)
{
    G_OBJECT_CLASS(le_main_window_parent_class)->constructed(object);
}

static void
le_main_window_class_init(LEMainWindowClass *klass)
{
    G_OBJECT_CLASS(klass)->constructed = le_main_window_constructed;
}

static void
le_main_window_init(LEMainWindow *self)
{
    self->drawing_area = gtk_drawing_area_new();
    gtk_widget_set_size_request(self->drawing_area, 112, 112);

    GtkWidget *grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 2);
    gtk_grid_set_column_spacing(GTK_GRID(grid), 2);
    gtk_grid_attach(GTK_GRID(grid), gtk_label_new("Set:"), 0, 0, 1, 1);
    self->set_selection_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(self->set_selection_combo), "Train");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(self->set_selection_combo), "Test");
    gtk_combo_box_set_active(GTK_COMBO_BOX(self->set_selection_combo), 1);
    gtk_grid_attach(GTK_GRID(grid), self->set_selection_combo, 1, 0, 1, 1);
    gtk_grid_attach(GTK_GRID(grid), gtk_label_new("Index:"), 0, 1, 1, 1);
    self->index_spin_button = gtk_spin_button_new_with_range(0, 59999, 1);
    gtk_grid_attach(GTK_GRID(grid), self->index_spin_button, 1, 1, 1, 1);
    gtk_grid_attach(GTK_GRID(grid), gtk_label_new("Label:"), 0, 2, 1, 1);
    self->label_entry = gtk_entry_new();
    g_object_set(self->label_entry, "editable", FALSE, NULL);
    gtk_grid_attach(GTK_GRID(grid), self->label_entry, 1, 2, 1, 1);

    GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 2);
    gtk_box_pack_start(GTK_BOX(hbox), self->drawing_area, TRUE, TRUE, 2);
    gtk_box_pack_start(GTK_BOX(hbox), gtk_separator_new(GTK_ORIENTATION_VERTICAL), FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(hbox), grid, FALSE, FALSE, 2);
    
    gtk_container_add(GTK_CONTAINER(self), hbox);

    g_action_map_add_action_entries(G_ACTION_MAP(self), win_entries, G_N_ELEMENTS(win_entries), self);
}

GtkWidget *
le_main_window_new(GtkApplication *application)
{
    LEMainWindow *window = g_object_new(LE_TYPE_MAIN_WINDOW, "application", application, NULL);
    return GTK_WIDGET(window);
}