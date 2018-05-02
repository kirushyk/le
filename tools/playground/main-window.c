/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "main-window.h"
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
    GtkWidget *epoch_label;
    
    gboolean dark;
    
    LeTrainingData *trainig_data;
    LeLogisticClassifier *classifier;
    
    cairo_surface_t *classifier_visualisation;
    
};

G_DEFINE_TYPE(LEMainWindow, le_main_window, GTK_TYPE_APPLICATION_WINDOW);

static gboolean
draw_callback(GtkWidget *widget, cairo_t *cr, gpointer data)
{
    gint i;
    guint width, height;
    LEMainWindow *window;
    
    width = gtk_widget_get_allocated_width(widget);
    height = gtk_widget_get_allocated_height(widget);
    window = LE_MAIN_WINDOW(data);
    
    window->dark ? cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 1.0) : cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 1.0);
    cairo_rectangle(cr, 0, 0, width, height);
    cairo_fill(cr);
  
    if (window->classifier_visualisation)
    {
        cairo_set_source_surface(cr, window->classifier_visualisation, 0, 0);
        cairo_paint(cr);
    }
    
    if (window->trainig_data)
    {
        LeMatrix *input = le_training_data_get_input(window->trainig_data);
        LeMatrix *output = le_training_data_get_output(window->trainig_data);;
        gint examples_count = le_matrix_get_width(input);
        for (i = 0; i < examples_count; i++)
        {
            // window->dark ? cairo_set_source_rgba (cr, 0.0, 0.0, 0.0, 1.0) : cairo_set_source_rgba (cr, 1.0, 1.0, 1.0, 1.0);
            double x = width * 0.5 + height * 0.5 * le_matrix_at(input, 0, i);
            double y = height * 0.5 - height * 0.5 * le_matrix_at(input, 1, i);
            window->dark ? cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 1.0) : cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 1.0);
            cairo_set_line_width(cr, 0.5);
            cairo_arc(cr, x, y, 2., 0., 2 * M_PI);
            if (le_matrix_at(output, 0, i) > 0.5)
            {
                cairo_fill(cr);
            }
            else
            {
                cairo_stroke(cr);
            }
        }
    }
    
    return FALSE;
}

static void
generate_activated(GSimpleAction *action, GVariant *parameter, gpointer data)
{
    LEMainWindow *window = LE_MAIN_WINDOW(data);
    guint width, height;
    
    width = gtk_widget_get_allocated_width(GTK_WIDGET(window->drawing_area));
    height = gtk_widget_get_allocated_height(GTK_WIDGET(window->drawing_area));
    
    window->trainig_data = pg_generate_data(g_variant_get_string(parameter, NULL));
    window->classifier = le_logistic_classifier_new_train(le_training_data_get_input(window->trainig_data), le_training_data_get_output(window->trainig_data), 1);
    
    if (window->classifier_visualisation)
    {
        cairo_surface_destroy(window->classifier_visualisation);
    }
    window->classifier_visualisation = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
    
    cairo_surface_flush(window->classifier_visualisation);
    guint8 *pixmap = cairo_image_surface_get_data(window->classifier_visualisation);
    for (gint y = 0; y < height; y++)
    {
        LeMatrix *row = le_matrix_new_uninitialized(2, width);
        for (gint x = 0; x < width; x++)
        {
            le_matrix_set_element(row, 0, x, x * 2.0f / width - 1.0f);
            le_matrix_set_element(row, 1, x, y * -2.0f / height + 1.0f);
        }
        
        LeMatrix *prediction = le_logistic_classifier_predict(window->classifier, row);
        
        le_matrix_free(row);
        
        for (gint x = 0; x < width; x++)
        {
            ARGB32 color = color_for_logistic(le_matrix_at(prediction, 0, x));
            ((ARGB32 *)pixmap)[y * width + x] = color;
        }
    }
    cairo_surface_mark_dirty(window->classifier_visualisation);
    gtk_widget_queue_draw(GTK_WIDGET(window));
}

static void
style_activated(GSimpleAction *action, GVariant *parameter, gpointer data)
{
    LEMainWindow *window = LE_MAIN_WINDOW(data);
    g_assert(LE_IS_MAIN_WINDOW(window));
    const gchar *style = g_variant_get_string(parameter, NULL);
    window->dark = 0 == g_strcmp0(style, "dark");
    gtk_widget_queue_draw(GTK_WIDGET(window));
}

static void
view_activated(GSimpleAction *action, GVariant *parameter, gpointer data)
{
    LEMainWindow *window = LE_MAIN_WINDOW(data);
    g_assert(LE_IS_MAIN_WINDOW(window));
    gtk_widget_queue_draw(GTK_WIDGET(window));
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
    { "style", style_activated, "s" },
    { "gen", generate_activated, "s" },
    { "style", style_activated, "s" },
    { "view", view_activated, "s", "\"q\"" },
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
    self->dark = FALSE;
    self->trainig_data = NULL;
    self->classifier = NULL;
    self->classifier_visualisation = NULL;
    
    GtkWidget *reset = gtk_button_new_from_icon_name("go-first", GTK_ICON_SIZE_LARGE_TOOLBAR);
    GtkWidget *start = gtk_button_new_from_icon_name("media-playback-start", GTK_ICON_SIZE_LARGE_TOOLBAR);
    GtkWidget *stop = gtk_button_new_from_icon_name("media-playback-stop", GTK_ICON_SIZE_LARGE_TOOLBAR);
    GtkWidget *step = gtk_button_new_from_icon_name("go-next", GTK_ICON_SIZE_LARGE_TOOLBAR);

    GtkWidget *learning_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 2);
    gtk_box_pack_start(GTK_BOX(learning_hbox), reset, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(learning_hbox), start, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(learning_hbox), stop, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(learning_hbox), step, FALSE, FALSE, 2);
    GtkWidget *learning_grid = gtk_grid_new();
    gtk_grid_set_column_spacing(GTK_GRID(learning_grid), 8);
    gtk_grid_attach(GTK_GRID(learning_grid), gtk_label_new("Epoch"), 0, 0, 1, 1);
    self->epoch_label = gtk_label_new("<big>0</big>");
    gtk_label_set_use_markup(GTK_LABEL(self->epoch_label), TRUE);
    gtk_grid_attach(GTK_GRID(learning_grid), self->epoch_label, 0, 1, 1, 1);
    gtk_grid_attach(GTK_GRID(learning_grid), gtk_label_new("Learning Rate"), 1, 0, 1, 1);
    GtkWidget *alpha_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(alpha_combo), "0.001");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(alpha_combo), "0.003");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(alpha_combo), "0.01");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(alpha_combo), "0.03");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(alpha_combo), "0.1");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(alpha_combo), "0.3");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(alpha_combo), "1");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(alpha_combo), "3");
    gtk_combo_box_set_active(GTK_COMBO_BOX(alpha_combo), 4);
    gtk_grid_attach(GTK_GRID(learning_grid), alpha_combo, 1, 1, 1, 1);
    gtk_grid_attach(GTK_GRID(learning_grid), gtk_label_new("Regularization"), 2, 0, 1, 1);
    GtkWidget *regularization_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(regularization_combo), "None");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(regularization_combo), "L1");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(regularization_combo), "L2");
    gtk_combo_box_set_active(GTK_COMBO_BOX(regularization_combo), 0);
    gtk_grid_attach(GTK_GRID(learning_grid), regularization_combo, 2, 1, 1, 1);
    gtk_grid_attach(GTK_GRID(learning_grid), gtk_label_new("Regularization Rate"), 3, 0, 1, 1);
    GtkWidget *lambda_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(lambda_combo), "0");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(lambda_combo), "0.001");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(lambda_combo), "0.003");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(lambda_combo), "0.01");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(lambda_combo), "0.03");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(lambda_combo), "0.1");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(lambda_combo), "0.3");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(lambda_combo), "1");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(lambda_combo), "3");
    gtk_combo_box_set_active(GTK_COMBO_BOX(lambda_combo), 0);
    gtk_grid_attach(GTK_GRID(learning_grid), lambda_combo, 3, 1, 1, 1);
    gtk_box_pack_start(GTK_BOX(learning_hbox), learning_grid, FALSE, FALSE, 2);
    
    GtkWidget *data_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    GtkWidget *label = gtk_label_new("<b>DATA</b>");
    gtk_label_set_use_markup(GTK_LABEL(label), TRUE);
    GtkWidget *rand_rb = gtk_radio_button_new_with_label(NULL, "Random");
    GtkWidget *linsep_rb = gtk_radio_button_new_with_label_from_widget(GTK_RADIO_BUTTON(rand_rb), "Linearly Separable");
    GtkWidget *nested_rb = gtk_radio_button_new_with_label_from_widget(GTK_RADIO_BUTTON(rand_rb), "Nested Circles");
    GtkWidget *svb_rb = gtk_radio_button_new_with_label_from_widget(GTK_RADIO_BUTTON(rand_rb), "SV Border");
    GtkWidget *spiral_rb = gtk_radio_button_new_with_label_from_widget(GTK_RADIO_BUTTON(rand_rb), "Spiral");
    GtkWidget *generate = gtk_button_new_with_label("Generate");
    gtk_box_pack_start(GTK_BOX(data_vbox), label, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(data_vbox), rand_rb, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(data_vbox), linsep_rb, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(data_vbox), nested_rb, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(data_vbox), svb_rb, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(data_vbox), spiral_rb, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(data_vbox), generate, FALSE, FALSE, 2);
    
    GtkWidget *model_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    label = gtk_label_new("<b>MODEL</b>");
    gtk_label_set_use_markup(GTK_LABEL(label), TRUE);
    gtk_box_pack_start(GTK_BOX(model_vbox), label, FALSE, FALSE, 2);
    GtkWidget *model_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo), "Logistic Unit");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo), "Support Vector Machine");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo), "Neural Network");
    gtk_combo_box_set_active(GTK_COMBO_BOX(model_combo), 0);
    gtk_box_pack_start(GTK_BOX(model_vbox), model_combo, FALSE, FALSE, 2);
    
    self->drawing_area = gtk_drawing_area_new();
    gtk_widget_set_size_request(self->drawing_area, 256, 256);
    g_signal_connect(G_OBJECT(self->drawing_area), "draw", G_CALLBACK(draw_callback), self);
    
    GtkWidget *output_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    label = gtk_label_new("<b>OUTPUT</b>");
    gtk_label_set_use_markup(GTK_LABEL(label), TRUE);
    gtk_box_pack_start(GTK_BOX(output_vbox), label, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(output_vbox), self->drawing_area, FALSE, FALSE, 2);
    
    GtkWidget *main_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 2);
    gtk_box_pack_start(GTK_BOX(main_hbox), data_vbox, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(main_hbox), gtk_separator_new(GTK_ORIENTATION_VERTICAL), FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(main_hbox), model_vbox, TRUE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(main_hbox), gtk_separator_new(GTK_ORIENTATION_VERTICAL), FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(main_hbox), output_vbox, FALSE, FALSE, 2);

    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    gtk_box_pack_start(GTK_BOX(vbox), learning_hbox, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(vbox), gtk_separator_new(GTK_ORIENTATION_HORIZONTAL), FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(vbox), main_hbox, FALSE, FALSE, 2);
    
    gtk_container_add(GTK_CONTAINER(self), vbox);
    g_action_map_add_action_entries(G_ACTION_MAP(self), win_entries, G_N_ELEMENTS(win_entries), self);
}

GtkWidget *
le_main_window_new (GtkApplication *application)
{
    LEMainWindow *window = g_object_new(LE_TYPE_MAIN_WINDOW, "application", application, NULL);
    gtk_window_set_default_size(GTK_WINDOW(window), 256, 256);
    return GTK_WIDGET(window);
}

// cairo_surface_destroy(window->classifier_visualisation);
