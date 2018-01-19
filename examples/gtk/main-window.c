#include "main-window.h"
#include <stdlib.h>
#include <le/le.h>

#define LE_TYPE_MAIN_WINDOW le_main_window_get_type ()
G_DECLARE_FINAL_TYPE (LEMainWindow, le_main_window, LE, MAIN_WINDOW, GtkApplicationWindow);

struct _LEMainWindow
{
    GtkApplicationWindow parent_instance;
    
    GtkWidget *drawing_area;
    
    gboolean dark;
    LeTrainingData *trainig_data;
    
};

G_DEFINE_TYPE (LEMainWindow, le_main_window, GTK_TYPE_APPLICATION_WINDOW);

static gboolean
draw_callback (GtkWidget *widget, cairo_t *cr, gpointer data)
{
    gint i;
    guint width, height;
    LEMainWindow *window;
    
    width = gtk_widget_get_allocated_width (widget);
    height = gtk_widget_get_allocated_height (widget);
    window = LE_MAIN_WINDOW (data);
    
    
    /*
    window->dark ? cairo_set_source_rgba (cr, 0.0, 0.0, 0.0, 1.0) : cairo_set_source_rgba (cr, 1.0, 1.0, 1.0, 1.0);
    cairo_rectangle (cr, 0, 0, width, height);
    cairo_fill(cr);
    */

    cairo_surface_t *surf = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
    gint half = (width + height) / 2;
    gint stride = cairo_image_surface_get_stride(surf);
    cairo_surface_flush (surf);
    guint8 *pixmap = cairo_image_surface_get_data(surf);
    for (gint y = 0; y < height; y++)
    {
        for (gint x = 0; x < width; x++)
        {
            if (x + y > half)
            {
                pixmap[y * stride + x * 4 + 2] = 32;
                pixmap[y * stride + x * 4 + 1] = 96;
            }
            else
            {
                pixmap[y * stride + x * 4 + 2] = 128;
                pixmap[y * stride + x * 4 + 1] = 96;
            }
            pixmap[y * stride + x * 4 + 3] = 255;
        }
    }
    cairo_surface_mark_dirty(surf);
    cairo_set_source_surface(cr, surf, 0, 0);
    cairo_paint(cr);

    cairo_surface_destroy(surf);
    
    return FALSE;
}

static void
generate_activated (GSimpleAction *action, GVariant *parameter, gpointer data)
{
    LEMainWindow *window = LE_MAIN_WINDOW (data);
    LeMatrix *x = le_matrix_new_rand(2, 32);
    LeMatrix *y = le_matrix_new_zeros(1, 32);
    window->trainig_data = le_training_data_new_take(x, y);
    
    printf("training data generated\n");
}

static void
style_activated (GSimpleAction *action, GVariant *parameter, gpointer data)
{
    LEMainWindow *window = LE_MAIN_WINDOW (data);
    g_assert (LE_IS_MAIN_WINDOW (window));
    const gchar *style = g_variant_get_string (parameter, NULL);
    window->dark = (g_strcmp0(style, "dark") == 0);
    
    /// @fixme: Only redraw drawing area only
    gtk_widget_queue_draw (GTK_WIDGET (window));
}

static void
view_activated (GSimpleAction *action, GVariant *parameter, gpointer data)
{
    LEMainWindow *window = LE_MAIN_WINDOW (data);
    g_assert (LE_IS_MAIN_WINDOW (window));
    const gchar *style = g_variant_get_string (parameter, NULL);
    /*
    if ((g_strcmp0(style, "q") == 0))
        window->projection = LE_PROJECTION_TIMEX;
    else if */
    
    gtk_widget_queue_draw (GTK_WIDGET (window));
}

static void
close_activated (GSimpleAction *action, GVariant *parameter, gpointer data)
{
    GtkWidget *window = GTK_WIDGET (data);
    g_object_set (window, "application", NULL, NULL);
    gtk_window_close (GTK_WINDOW (window));
}

static GActionEntry win_entries[] =
{
    { "style", style_activated, "s" },
    { "gen", generate_activated, "s" },
    { "style", style_activated, "s" },
    { "view", view_activated, "s", "\"q\"" },
    { "close", close_activated }
};

static void
le_main_window_constructed (GObject *object)
{
    G_OBJECT_CLASS (le_main_window_parent_class)->constructed (object);
}

static void
le_main_window_class_init (LEMainWindowClass *klass)
{
    gint i;
    GObjectClass *object_class = G_OBJECT_CLASS (klass);
    
    object_class->constructed = le_main_window_constructed;
    
}

static void
le_main_window_init (LEMainWindow *self)
{
    self->dark = FALSE;
    self->trainig_data = NULL;
    
    self->drawing_area = gtk_drawing_area_new ();
    gtk_widget_set_size_request (self->drawing_area, 640, 480);
    
    g_signal_connect (G_OBJECT (self->drawing_area), "draw", G_CALLBACK (draw_callback), self);
    
    gtk_container_add (GTK_CONTAINER (self), self->drawing_area);
    
    g_action_map_add_action_entries (G_ACTION_MAP (self), win_entries, G_N_ELEMENTS (win_entries), self);
}

GtkWidget *
le_main_window_new (GtkApplication *application)
{
    LEMainWindow *window;
    
    window = g_object_new (LE_TYPE_MAIN_WINDOW, "application", application, NULL);

    gtk_window_set_title (GTK_WINDOW (window), "Le Gtk+ Demo");
    gtk_window_set_default_size (GTK_WINDOW (window), 640, 480);
    
    return GTK_WIDGET (window);
}