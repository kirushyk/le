/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "ma-main-window.h"
#include <assert.h>
#include <stdlib.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>
#include <ext/mnist/lemnist.h>
#include <math.h>

#define CLASSES_COUNT 10
#define SCALE 8

#define LE_TYPE_MAIN_WINDOW le_main_window_get_type()
G_DECLARE_FINAL_TYPE(LEMainWindow, le_main_window, LE, MAIN_WINDOW, GtkApplicationWindow);

struct _LEMainWindow
{
    GtkApplicationWindow parent_instance;
    
    GtkWidget *drawing_area;
    MNIST *mnist;
    LeTensor *mean_inputs;
    guint32 index;
    
    cairo_surface_t *image_visualisation;
};

G_DEFINE_TYPE(LEMainWindow, le_main_window, GTK_TYPE_APPLICATION_WINDOW);

static cairo_surface_t *
render_image(guint8 *data)
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

    cairo_scale(cr, SCALE, SCALE);
    for (guint8 i = 0; i < 10; i++)
    {
        LeTensor *image = le_tensor_pick(window->mean_inputs, i);
        cairo_surface_t *image_visualisation = render_image(image->data);
        gdouble x = ((i < 5) ? i : (i - 5)) * 28;
        gdouble y = i < 5 ? 0 : 28;
        cairo_set_source_surface(cr, image_visualisation, x, y);
        cairo_surface_destroy(image_visualisation);
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
le_main_window_init(LEMainWindow *self)
{
    self->drawing_area = gtk_drawing_area_new();
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(self->drawing_area), draw_callback, self, NULL);
    gtk_widget_set_size_request(self->drawing_area, SCALE * 28 * 5, SCALE * 28 * 2);

    gtk_window_set_child(GTK_WINDOW(self), self->drawing_area);
    
    self->mnist = le_mnist_load(NULL);
    self->image_visualisation = NULL;

    LeTensor *mean_inputs_u32 = le_tensor_new_zeros (LE_TYPE_U32, le_shape_new(3, CLASSES_COUNT, 28, 28));
    guint32 examples_count = 60000;
    guint32 contrast = 5;
    for (guint32 i = 0; i < examples_count; i++)
    {
        LeTensor *current_image = le_tensor_pick(le_data_set_get_input(self->mnist->train), i);
        LeTensor *current_image_u32 = le_tensor_new_cast(current_image, LE_TYPE_U32);
        guint8 label = le_tensor_at_u8(le_data_set_get_output(self->mnist->train), i);
        LeTensor *mean_image_u32 = le_tensor_pick(mean_inputs_u32, label);
        le_tensor_add_tensor(mean_image_u32, current_image_u32);
        le_tensor_unref(current_image_u32);
    }
    self->mean_inputs = le_tensor_new_uninitialized (LE_TYPE_U8, le_shape_new(3, CLASSES_COUNT, 28, 28));
    for (guint32 i = 0; i < 10; i++)
    {
        LeTensor *current_mean_image_u32 = le_tensor_pick(mean_inputs_u32, i);
        le_tensor_div_u32(current_mean_image_u32, examples_count / contrast);
        LeTensor *mean_image_u8 = le_tensor_new_cast(current_mean_image_u32, LE_TYPE_U8);
        LeTensor *current_mean_image_u8 = le_tensor_pick(self->mean_inputs, i);
        le_tensor_assign(current_mean_image_u8, mean_image_u8);
    }
    le_tensor_unref(mean_inputs_u32);

    g_action_map_add_action_entries(G_ACTION_MAP(self), win_entries, G_N_ELEMENTS(win_entries), self);
}

GtkWidget *
le_main_window_new(GtkApplication *application)
{
    LEMainWindow *window = g_object_new(LE_TYPE_MAIN_WINDOW, "application", application, NULL);
    assert(LE_IS_MAIN_WINDOW(window));
    return GTK_WIDGET(window);
}
