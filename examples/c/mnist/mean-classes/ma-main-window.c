/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "ma-main-window.h"
#include <assert.h>
#include <stdlib.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>
#include <ext/mnist/lemnist.h>
#include <math.h>

#define LE_TYPE_MAIN_WINDOW le_main_window_get_type()
G_DECLARE_FINAL_TYPE(LEMainWindow, le_main_window, LE, MAIN_WINDOW, GtkApplicationWindow);

struct _LEMainWindow
{
    GtkApplicationWindow parent_instance;
    
    GtkWidget *drawing_area;
    GtkWidget *set_selection_combo;
    GtkWidget *index_spin_button;
    
    MNIST *data_set;
    LeTensor *input, *output;
    uint32_t index;
    
    cairo_surface_t *image_visualisation;
};

//if (self->image_visualisation)
//    cairo_surface_destroy(self->image_visualisation);

G_DEFINE_TYPE(LEMainWindow, le_main_window, GTK_TYPE_APPLICATION_WINDOW);

static cairo_surface_t *
render_image(uint8_t *data)
{
    if (data == NULL)
    {
        return cairo_image_surface_create(CAIRO_FORMAT_A8, 28, 28);
    }

    return cairo_image_surface_create_for_data(data, CAIRO_FORMAT_A8, 28, 28, 28);
}

static void
draw_callback(GtkDrawingArea *drawing_area, cairo_t *cr, int width, int height, gpointer data)
{
    LEMainWindow *window = LE_MAIN_WINDOW(data);
    
    if (window->image_visualisation)
    {
        cairo_scale(cr, 4.0, 4.0);
        cairo_set_source_surface(cr, window->image_visualisation, 0, 0);
        cairo_paint(cr);
    }
}

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
update_image(LEMainWindow *window)
{
    if (window->image_visualisation) {
        cairo_surface_destroy(window->image_visualisation);
        window->image_visualisation = NULL;
    }
    
    if (window->input) {
        LeTensor *image = le_tensor_pick(window->input, window->index);
        printf("index: %u\n", window->index);
        if (image) {
            window->image_visualisation = render_image(image->data);
        }
    }
    
    if (window->output) {
        int label = le_tensor_at_u8(window->output, window->index);
        char buffer[8];
        sprintf(buffer, "%d", label);
    }

    gtk_widget_queue_draw(GTK_WIDGET(window->drawing_area));
}

static void
set_changed(GtkComboBox *combo_box, gpointer data)
{
    LEMainWindow *window = LE_MAIN_WINDOW(data);
    
    if (window->data_set == NULL)
        return;

    if (window->data_set->train == NULL || window->data_set->test == NULL)
        return;
    
    if (gtk_combo_box_get_active(combo_box)) {
        window->input = le_data_set_get_input(window->data_set->test);
        window->output = le_data_set_get_output(window->data_set->test);
    } else {
        window->input = le_data_set_get_input(window->data_set->train);
        window->output = le_data_set_get_output(window->data_set->train);
    }

    if (window->output == NULL)
        return;

    int test_examples_count = le_shape_get_elements_count(window->output->shape);
    if (window->index >= test_examples_count) {
        window->index = test_examples_count - 1;
    }

    gtk_spin_button_set_range(GTK_SPIN_BUTTON(window->index_spin_button), 0, test_examples_count - 1);
    
    update_image(window);
}

static void
index_changed(GtkSpinButton *spin_button, gpointer data)
{
    LEMainWindow *window = LE_MAIN_WINDOW(data);
        
    window->index = (uint32_t)gtk_spin_button_get_value(spin_button);
    
    update_image(window);
}

static void
le_main_window_init(LEMainWindow *self)
{
    self->drawing_area = gtk_drawing_area_new();
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(self->drawing_area), draw_callback, self, NULL);
    gtk_widget_set_size_request(self->drawing_area, 112, 112);

    GtkWidget *grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 2);
    gtk_grid_set_column_spacing(GTK_GRID(grid), 2);
    gtk_grid_attach(GTK_GRID(grid), gtk_label_new("Set:"), 0, 0, 1, 1);
    self->set_selection_combo = gtk_combo_box_text_new();
    g_signal_connect(G_OBJECT(self->set_selection_combo), "changed", G_CALLBACK(set_changed), self);
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(self->set_selection_combo), "Train");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(self->set_selection_combo), "Test");
    gtk_combo_box_set_active(GTK_COMBO_BOX(self->set_selection_combo), 0);
    gtk_grid_attach(GTK_GRID(grid), self->set_selection_combo, 1, 0, 1, 1);
    gtk_grid_attach(GTK_GRID(grid), gtk_label_new("Class:"), 0, 1, 1, 1);
    self->index_spin_button = gtk_spin_button_new_with_range(0, 9, 1);
    g_signal_connect(G_OBJECT(self->index_spin_button), "value-changed", G_CALLBACK(index_changed), self);
    gtk_grid_attach(GTK_GRID(grid), self->index_spin_button, 1, 1, 1, 1);

    GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 2);
    gtk_box_append(GTK_BOX(hbox), self->drawing_area);
    gtk_box_append(GTK_BOX(hbox), gtk_separator_new(GTK_ORIENTATION_VERTICAL));
    gtk_box_append(GTK_BOX(hbox), grid);
    
    gtk_window_set_child(GTK_WINDOW(self), hbox);
    
    self->data_set = le_mnist_load(NULL);
    self->image_visualisation = NULL;
    self->input = NULL;
    self->output = NULL;
    index_changed(GTK_SPIN_BUTTON(self->index_spin_button), self);
    set_changed(GTK_COMBO_BOX(self->set_selection_combo), self);

    g_action_map_add_action_entries(G_ACTION_MAP(self), win_entries, G_N_ELEMENTS(win_entries), self);
}

GtkWidget *
le_main_window_new(GtkApplication *application)
{
    LEMainWindow *window = g_object_new(LE_TYPE_MAIN_WINDOW, "application", application, NULL);
    assert(LE_IS_MAIN_WINDOW(window));
    return GTK_WIDGET(window);
}
