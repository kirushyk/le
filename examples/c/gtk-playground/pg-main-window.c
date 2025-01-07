/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "pg-main-window.h"
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <glib.h>
#include <le/le.h>
#include <ext/simple-dataset/ledataset.h>
#include "pg-color.h"
#include "pg-generate-data.h"

#define LE_TYPE_MAIN_WINDOW le_main_window_get_type ()
G_DECLARE_FINAL_TYPE (LEMainWindow, le_main_window, LE, MAIN_WINDOW, GtkApplicationWindow);

struct _LEMainWindow
{
  GtkApplicationWindow parent_instance;

  GtkWidget *drawing_area;
  GtkWidget *epoch_label;

  gboolean dark;

  LeDataSet *train_data;
  LeDataSet *test_data;
  LeModel *model;
  LeOptimizer *optimizer;

  cairo_surface_t *classifier_visualisation;

  GtkWidget *rand_rb;
  GtkWidget *linsep_rb;
  GtkWidget *nested_rb;
  GtkWidget *svb_rb;
  GtkWidget *spiral_rb;
  GtkWidget *train_set_size_drop_down;
  GtkWidget *test_set_drop_down;

  GtkWidget *gd_vbox;
  GtkWidget *pr_vbox;
  GtkWidget *svm_vbox;
  GtkWidget *knn_vbox;
  GtkWidget *svm_kernel_drop_down;
  GtkWidget *svm_c_drop_down;
  GtkWidget *knn_k_drop_down;

  GtkWidget *polynomia_degree_drop_down;
  GtkWidget *alpha_drop_down;
  GtkWidget *regularization_drop_down;
  GtkWidget *lambda_drop_down;

  PreferredModelType preferred_model_type;
};

G_DEFINE_TYPE (LEMainWindow, le_main_window, GTK_TYPE_APPLICATION_WINDOW);

static void
draw_callback (GtkDrawingArea *drawing_area, cairo_t *cr, int width, int height, gpointer data)
{
  LEMainWindow *window = LE_MAIN_WINDOW (data);

  window->dark ? cairo_set_source_rgba (cr, 0.0, 0.0, 0.0, 1.0) : cairo_set_source_rgba (cr, 1.0, 1.0, 1.0, 1.0);
  cairo_rectangle (cr, 0, 0, width, height);
  cairo_fill (cr);

  if (window->classifier_visualisation) {
    cairo_set_source_surface (cr, window->classifier_visualisation, 0, 0);
    cairo_paint (cr);
  }

  if (window->train_data) {
    LeTensor *input = le_data_set_get_input (window->train_data);
    LeTensor *output = le_data_set_get_output (window->train_data);
    gint examples_count = le_matrix_get_width (input);
    for (gint i = 0; i < examples_count; i++) {
      gdouble x = width * 0.5 + height * 0.5 * le_matrix_at_f32 (input, 0, i);
      gdouble y = height * 0.5 - height * 0.5 * le_matrix_at_f32 (input, 1, i);
      cairo_set_line_width (cr, 0.5);
      cairo_arc (cr, x, y, 2., 0., 2 * G_PI);
      le_matrix_at_f32 (output, 0, i) > 0.5 ? cairo_set_source_rgba (cr, 1.0, 0.5, 0.0, 1.0)
                                            : cairo_set_source_rgba (cr, 0.0, 0.5, 1.0, 1.0);
      cairo_fill (cr);
      cairo_arc (cr, x, y, 2., 0., 2 * G_PI);
      window->dark ? cairo_set_source_rgba (cr, 1.0, 1.0, 1.0, 1.0) : cairo_set_source_rgba (cr, 0.0, 0.0, 0.0, 1.0);
      cairo_stroke (cr);
    }
  }

  if (window->test_data) {
    LeTensor *input = le_data_set_get_input (window->test_data);
    LeTensor *output = le_data_set_get_output (window->test_data);
    gint examples_count = le_matrix_get_width (input);
    for (gint i = 0; i < examples_count; i++) {
      gdouble x = width * 0.5 + height * 0.5 * le_matrix_at_f32 (input, 0, i);
      gdouble y = height * 0.5 - height * 0.5 * le_matrix_at_f32 (input, 1, i);
      window->dark ? cairo_set_source_rgba (cr, 0.0, 1.0, 0.0, 1.0) : cairo_set_source_rgba (cr, 0.0, 0.5, 0.0, 1.0);
      cairo_set_line_width (cr, 0.5);
      cairo_arc (cr, x, y, 2., 0., 2 * G_PI);
      le_matrix_at_f32 (output, 0, i) > 0.5 ? cairo_set_source_rgba (cr, 1.0, 0.5, 0.0, 1.0)
                                            : cairo_set_source_rgba (cr, 0.0, 0.5, 1.0, 1.0);
      cairo_stroke (cr);
    }
  }
}

static cairo_surface_t *
render_predictions (LeModel *model, guint width, guint height)
{
  cairo_surface_t *surface = cairo_image_surface_create (CAIRO_FORMAT_ARGB32, width, height);

  if (model == NULL) {
    return surface;
  }

  cairo_surface_flush (surface);
  guint8 *pixmap = cairo_image_surface_get_data (surface);
  for (gint y = 0; y < height; y++) {
    LeTensor *row = le_matrix_new_uninitialized (LE_TYPE_FLOAT32, 2, width);
    for (gint x = 0; x < width; x++) {
      le_matrix_set (row, 0, x, x * 2.0f / width - 1.0f);
      le_matrix_set (row, 1, x, y * -2.0f / height + 1.0f);
    }

    LeTensor *prediction = le_model_predict (model, row);

    le_tensor_free (row);

    if (prediction != NULL) {
      for (gint x = 0; x < width; x++) {
        BGRA32 color = color_for_logistic (le_matrix_at_f32 (prediction, 0, x));
        ((BGRA32 *)pixmap)[y * width + x] = color;
      }
    }
  }
  cairo_surface_mark_dirty (surface);
  return surface;
}

void
erase_model (LEMainWindow *self)
{
  if (self->model)
    g_object_unref (self->model);
  self->model = NULL;
  if (self->optimizer)
    g_object_unref (self->optimizer);
  self->optimizer = NULL;
  if (self->classifier_visualisation)
    cairo_surface_destroy (self->classifier_visualisation);
  self->classifier_visualisation = NULL;
}

static void
update_epoch_label (LEMainWindow *self)
{
  unsigned epoch = 0;
  if (self->optimizer) {
    // epoch = LE_OPTIMIZER(self->optimizer)->epoch;
  }
  gchar buffer[16];
  sprintf (buffer, "%u", epoch);
  gtk_label_set_text (GTK_LABEL (self->epoch_label), buffer);
}

void
train_current_model (LEMainWindow *self)
{
  if (self->train_data == NULL)
    return;

  gfloat learning_rate = 1.0f;
  const gchar *learning_rate_str = gtk_string_object_get_string (
      (GtkStringObject *)gtk_drop_down_get_selected_item (GTK_DROP_DOWN (self->alpha_drop_down)));
  setlocale (LC_NUMERIC, "C");
  sscanf (learning_rate_str, "%f", &learning_rate);

  switch (self->preferred_model_type) {
  case PREFERRED_MODEL_TYPE_SUPPORT_VECTOR_MACHINE: {
    LeSVMTrainingOptions options;
    switch (gtk_drop_down_get_selected (GTK_DROP_DOWN (self->svm_kernel_drop_down))) {
    case 1:
      options.kernel = LE_KERNEL_RBF;
      break;
    case 0:
    default:
      options.kernel = LE_KERNEL_LINEAR;
      break;
    }
    sscanf (gtk_string_object_get_string (
                (GtkStringObject *)gtk_drop_down_get_selected_item (GTK_DROP_DOWN (self->svm_c_drop_down))),
        "%f", &options.c);
    LeTensor *labels = le_tensor_new_copy (le_data_set_get_output (self->train_data));
    le_tensor_apply_sgn (labels);
    le_svm_train (LE_SVM (self->model), le_data_set_get_input (self->train_data), labels, options);
    le_tensor_free (labels);
  } break;

  case PREFERRED_MODEL_TYPE_NEURAL_NETWORK: {
    if (self->optimizer == NULL) {
      LeTensor *x = le_data_set_get_input (self->train_data);
      LeTensor *labels = le_tensor_new_copy (le_data_set_get_output (self->train_data));
      self->optimizer = LE_OPTIMIZER (le_bgd_new (self->model, x, labels, learning_rate));
    }
    le_optimizer_set_learning_rate (self->optimizer, learning_rate);
    for (unsigned i = 0; i <= 400; i++) {
      le_optimizer_step (LE_OPTIMIZER (self->optimizer));
    }
    /// g_object_unref(LE_BGD(self->optimizer));
    /// @todo: g_free
    /// le_tensor_free(labels);
  } break;

  case PREFERRED_MODEL_TYPE_KNN: {
    unsigned k = atoi (gtk_string_object_get_string (
        (GtkStringObject *)gtk_drop_down_get_selected_item (GTK_DROP_DOWN (self->knn_k_drop_down))));
    le_knn_train (
        LE_KNN (self->model), le_data_set_get_input (self->train_data), le_data_set_get_output (self->train_data), k);
  } break;

  case PREFERRED_MODEL_TYPE_POLYNOMIAL_REGRESSION:
  default: {
    LeLogisticClassifierTrainingOptions options;
    options.max_iterations = 400;
    options.polynomia_degree = atoi (gtk_string_object_get_string (
        (GtkStringObject *)gtk_drop_down_get_selected_item (GTK_DROP_DOWN (self->polynomia_degree_drop_down))));
    options.learning_rate = learning_rate;
    switch (gtk_drop_down_get_selected (GTK_DROP_DOWN (self->regularization_drop_down))) {
    case 1:
      options.regularization = LE_REGULARIZATION_L1;
      break;
    case 2:
      options.regularization = LE_REGULARIZATION_L2;
      break;
    case 0:
    default:
      options.regularization = LE_REGULARIZATION_NONE;
      break;
    }
    sscanf (gtk_string_object_get_string (
                (GtkStringObject *)gtk_drop_down_get_selected_item (GTK_DROP_DOWN (self->lambda_drop_down))),
        "%f", &options.lambda);
    le_logistic_classifier_train (LE_LOGISTIC_CLASSIFIER (self->model), le_data_set_get_input (self->train_data),
        le_data_set_get_output (self->train_data), options);
  } break;
  }

  self->classifier_visualisation = render_predictions (self->model,
      gtk_widget_get_width (GTK_WIDGET (self->drawing_area)), gtk_widget_get_height (GTK_WIDGET (self->drawing_area)));

  gtk_widget_queue_draw (GTK_WIDGET (self->drawing_area));

  update_epoch_label (self);
}

void
create_model (LEMainWindow *self)
{
  erase_model (self);

  switch (self->preferred_model_type) {
  case PREFERRED_MODEL_TYPE_SUPPORT_VECTOR_MACHINE:
    self->model = (LeModel *)le_svm_new ();
    break;

  case PREFERRED_MODEL_TYPE_NEURAL_NETWORK:
    self->model = (LeModel *)le_sequential_new ();
    le_sequential_add (LE_SEQUENTIAL (self->model), LE_LAYER (le_dense_layer_new ("D1", 2, 30)));
    le_sequential_add (LE_SEQUENTIAL (self->model), LE_LAYER (le_activation_layer_new ("A1", LE_ACTIVATION_TANH)));
    le_sequential_add (LE_SEQUENTIAL (self->model), LE_LAYER (le_dense_layer_new ("D2", 30, 50)));
    le_sequential_add (LE_SEQUENTIAL (self->model), LE_LAYER (le_activation_layer_new ("A2", LE_ACTIVATION_TANH)));
    le_sequential_add (LE_SEQUENTIAL (self->model), LE_LAYER (le_dense_layer_new ("D3", 50, 30)));
    le_sequential_add (LE_SEQUENTIAL (self->model), LE_LAYER (le_activation_layer_new ("A3", LE_ACTIVATION_TANH)));
    le_sequential_add (LE_SEQUENTIAL (self->model), LE_LAYER (le_dense_layer_new ("D4", 30, 1)));
    le_sequential_add (LE_SEQUENTIAL (self->model), LE_LAYER (le_activation_layer_new ("A4", LE_ACTIVATION_SIGMOID)));
    le_sequential_set_loss (LE_SEQUENTIAL (self->model), LE_LOSS_LOGISTIC);
    break;

  case PREFERRED_MODEL_TYPE_KNN:
    self->model = (LeModel *)le_knn_new ();
    break;

  case PREFERRED_MODEL_TYPE_POLYNOMIAL_REGRESSION:
  default:
    self->model = (LeModel *)le_logistic_classifier_new ();
    break;
  }
}

static void
generate_data (LEMainWindow *self, const gchar *pattern)
{
  unsigned examples_count = atoi (gtk_string_object_get_string (
      (GtkStringObject *)gtk_drop_down_get_selected_item (GTK_DROP_DOWN (self->train_set_size_drop_down))));
  self->train_data = pg_generate_data (pattern, examples_count);
  examples_count = atoi (gtk_string_object_get_string (
      (GtkStringObject *)gtk_drop_down_get_selected_item (GTK_DROP_DOWN (self->test_set_drop_down))));
  self->test_data = pg_generate_data (pattern, examples_count);

  gtk_widget_queue_draw (GTK_WIDGET (self->drawing_area));
}

static void
generate_menu_activated (GSimpleAction *action, GVariant *parameter, gpointer data)
{
  LEMainWindow *window = LE_MAIN_WINDOW (data);
  generate_data (window, g_variant_get_string (parameter, NULL));
}

static void
style_activated (GSimpleAction *action, GVariant *parameter, gpointer data)
{
  LEMainWindow *window = LE_MAIN_WINDOW (data);
  g_assert (LE_IS_MAIN_WINDOW (window));
  const gchar *style = g_variant_get_string (parameter, NULL);
  window->dark = 0 == g_strcmp0 (style, "dark");
  gtk_widget_queue_draw (GTK_WIDGET (window->drawing_area));
}

static void
view_activated (GSimpleAction *action, GVariant *parameter, gpointer data)
{
  LEMainWindow *window = LE_MAIN_WINDOW (data);
  g_assert (LE_IS_MAIN_WINDOW (window));
  gtk_widget_queue_draw (GTK_WIDGET (window->drawing_area));
}

static void
close_activated (GSimpleAction *action, GVariant *parameter, gpointer data)
{
  GtkWidget *window = GTK_WIDGET (data);
  g_object_set (window, "application", NULL, NULL);
  gtk_window_close (GTK_WINDOW (window));
}

static GActionEntry win_entries[] = { { "style", style_activated, "s" }, { "gen", generate_menu_activated, "s" },
  { "style", style_activated, "s" }, { "view", view_activated, "s", "\"q\"" }, { "close", close_activated } };

static void
le_main_window_constructed (GObject *object)
{
  G_OBJECT_CLASS (le_main_window_parent_class)->constructed (object);
}

static void
le_main_window_class_init (LEMainWindowClass *klass)
{
  G_OBJECT_CLASS (klass)->constructed = le_main_window_constructed;
}

static void
generate_button_clicked (GtkButton *button, gpointer user_data)
{
  LEMainWindow *window = LE_MAIN_WINDOW (user_data);
  if (gtk_check_button_get_active (GTK_CHECK_BUTTON (window->rand_rb)))
    generate_data (window, "rand");
  else if (gtk_check_button_get_active (GTK_CHECK_BUTTON (window->linsep_rb)))
    generate_data (window, "linsep");
  else if (gtk_check_button_get_active (GTK_CHECK_BUTTON (window->nested_rb)))
    generate_data (window, "nested");
  else if (gtk_check_button_get_active (GTK_CHECK_BUTTON (window->svb_rb)))
    generate_data (window, "svb");
  else
    generate_data (window, "spiral");
}

void
le_main_window_set_preffered_model (GtkWidget *window, PreferredModelType model_type)
{
  LEMainWindow *self = LE_MAIN_WINDOW (window);

  self->preferred_model_type = model_type;

  switch (self->preferred_model_type) {
  case PREFERRED_MODEL_TYPE_SUPPORT_VECTOR_MACHINE:
    gtk_widget_set_visible (self->svm_vbox, TRUE);
    gtk_widget_set_visible (self->pr_vbox, FALSE);
    gtk_widget_set_visible (self->gd_vbox, FALSE);
    gtk_widget_set_visible (self->knn_vbox, FALSE);
    break;

  case PREFERRED_MODEL_TYPE_NEURAL_NETWORK:
    gtk_widget_set_visible (self->svm_vbox, FALSE);
    gtk_widget_set_visible (self->pr_vbox, FALSE);
    gtk_widget_set_visible (self->gd_vbox, TRUE);
    gtk_widget_set_visible (self->knn_vbox, FALSE);
    break;

  case PREFERRED_MODEL_TYPE_KNN:
    gtk_widget_set_visible (self->svm_vbox, FALSE);
    gtk_widget_set_visible (self->pr_vbox, FALSE);
    gtk_widget_set_visible (self->gd_vbox, FALSE);
    gtk_widget_set_visible (self->knn_vbox, TRUE);
    break;

  case PREFERRED_MODEL_TYPE_POLYNOMIAL_REGRESSION:
  default:
    gtk_widget_set_visible (self->svm_vbox, FALSE);
    gtk_widget_set_visible (self->pr_vbox, TRUE);
    gtk_widget_set_visible (self->gd_vbox, TRUE);
    gtk_widget_set_visible (self->knn_vbox, FALSE);
    break;
  }

  create_model (self);
}

void
model_drop_down_changed (GtkDropDown *widget, gpointer user_data)
{
  LEMainWindow *self = LE_MAIN_WINDOW (user_data);

  switch (gtk_drop_down_get_selected (widget)) {
  case 1:
    le_main_window_set_preffered_model ((GtkWidget *)self, PREFERRED_MODEL_TYPE_SUPPORT_VECTOR_MACHINE);
    break;

  case 2:
    le_main_window_set_preffered_model ((GtkWidget *)self, PREFERRED_MODEL_TYPE_NEURAL_NETWORK);
    break;

  case 3:
    le_main_window_set_preffered_model ((GtkWidget *)self, PREFERRED_MODEL_TYPE_KNN);
    break;

  case 0:
  default:
    le_main_window_set_preffered_model ((GtkWidget *)self, PREFERRED_MODEL_TYPE_POLYNOMIAL_REGRESSION);
    break;
  }
}

static void
reset_button_clicked (GtkButton *button, gpointer user_data)
{
  LEMainWindow *self = LE_MAIN_WINDOW (user_data);

  create_model (self);
  if (self->optimizer) {
    g_object_unref (self->optimizer);
    self->optimizer = NULL;
  }
  gtk_widget_queue_draw (GTK_WIDGET (self->drawing_area));

  update_epoch_label (self);
}

static void
start_button_clicked (GtkButton *button, gpointer user_data)
{
  LEMainWindow *self = LE_MAIN_WINDOW (user_data);

  train_current_model (self);
}

static void
le_main_window_init (LEMainWindow *self)
{
  self->dark = FALSE;
  self->train_data = NULL;
  self->test_data = NULL;
  self->model = NULL;
  self->optimizer = NULL;
  self->classifier_visualisation = NULL;
  self->preferred_model_type = PREFERRED_MODEL_TYPE_POLYNOMIAL_REGRESSION;

  GtkWidget *reset = gtk_button_new_from_icon_name ("go-first");
  g_signal_connect (G_OBJECT (reset), "clicked", G_CALLBACK (reset_button_clicked), self);
  GtkWidget *start = gtk_button_new_from_icon_name ("media-playback-start");
  g_signal_connect (G_OBJECT (start), "clicked", G_CALLBACK (start_button_clicked), self);
  GtkWidget *stop = gtk_button_new_from_icon_name ("media-playback-stop");
  GtkWidget *step = gtk_button_new_from_icon_name ("go-next");

  GtkWidget *learning_hbox = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 2);
  gtk_box_append (GTK_BOX (learning_hbox), reset);
  gtk_box_append (GTK_BOX (learning_hbox), start);
  gtk_box_append (GTK_BOX (learning_hbox), stop);
  gtk_box_append (GTK_BOX (learning_hbox), step);
  GtkWidget *learning_grid = gtk_grid_new ();
  gtk_grid_set_column_spacing (GTK_GRID (learning_grid), 8);
  gtk_grid_attach (GTK_GRID (learning_grid), gtk_label_new ("Epoch"), 0, 0, 1, 1);
  self->epoch_label = gtk_label_new ("<big>0</big>");
  gtk_label_set_use_markup (GTK_LABEL (self->epoch_label), TRUE);
  gtk_grid_attach (GTK_GRID (learning_grid), self->epoch_label, 0, 1, 1, 1);
  gtk_box_append (GTK_BOX (learning_hbox), learning_grid);

  GtkWidget *data_vbox = gtk_box_new (GTK_ORIENTATION_VERTICAL, 2);
  GtkWidget *label = gtk_label_new ("<b>DATA</b>");
  gtk_label_set_use_markup (GTK_LABEL (label), TRUE);

  self->svb_rb = gtk_check_button_new_with_label ("Support Vectors");
  gtk_check_button_set_active (GTK_CHECK_BUTTON (self->svb_rb), TRUE);
  self->linsep_rb = gtk_check_button_new_with_label ("Linearly Separable");
  gtk_check_button_set_group (GTK_CHECK_BUTTON (self->linsep_rb), GTK_CHECK_BUTTON (self->svb_rb));
  self->nested_rb = gtk_check_button_new_with_label ("Nested Circles");
  gtk_check_button_set_group (GTK_CHECK_BUTTON (self->nested_rb), GTK_CHECK_BUTTON (self->svb_rb));
  self->spiral_rb = gtk_check_button_new_with_label ("Spiral");
  gtk_check_button_set_group (GTK_CHECK_BUTTON (self->spiral_rb), GTK_CHECK_BUTTON (self->svb_rb));
  self->rand_rb = gtk_check_button_new_with_label ("Random");
  gtk_check_button_set_group (GTK_CHECK_BUTTON (self->rand_rb), GTK_CHECK_BUTTON (self->svb_rb));

  const gchar *train_set_sizes[] = { "256", "128", "64", "32", "16", "8", NULL };
  self->train_set_size_drop_down = gtk_drop_down_new_from_strings (train_set_sizes);
  gtk_drop_down_set_selected (GTK_DROP_DOWN (self->train_set_size_drop_down), 1);

  const gchar *test_set_sizes[] = { "32", "16", "8", "4", NULL };
  self->test_set_drop_down = gtk_drop_down_new_from_strings (test_set_sizes);
  gtk_drop_down_set_selected (GTK_DROP_DOWN (self->test_set_drop_down), 2);

  GtkWidget *generate = gtk_button_new_with_label ("Generate");
  g_signal_connect (G_OBJECT (generate), "clicked", G_CALLBACK (generate_button_clicked), self);
  gtk_box_append (GTK_BOX (data_vbox), label);
  gtk_box_append (GTK_BOX (data_vbox), self->svb_rb);
  gtk_box_append (GTK_BOX (data_vbox), self->linsep_rb);
  gtk_box_append (GTK_BOX (data_vbox), self->nested_rb);
  gtk_box_append (GTK_BOX (data_vbox), self->spiral_rb);
  gtk_box_append (GTK_BOX (data_vbox), self->rand_rb);
  gtk_box_append (GTK_BOX (data_vbox), gtk_label_new ("Train Set Size"));
  gtk_box_append (GTK_BOX (data_vbox), self->train_set_size_drop_down);
  gtk_box_append (GTK_BOX (data_vbox), gtk_label_new ("Test Set Size"));
  gtk_box_append (GTK_BOX (data_vbox), self->test_set_drop_down);
  gtk_box_append (GTK_BOX (data_vbox), generate);

  GtkWidget *model_vbox = gtk_box_new (GTK_ORIENTATION_VERTICAL, 2);
  label = gtk_label_new ("<b>MODEL</b>");
  gtk_label_set_use_markup (GTK_LABEL (label), TRUE);
  gtk_box_append (GTK_BOX (model_vbox), label);
  const gchar *model_names[] = { "Polynomial Regression", "Support Vector Machine (SVM)", "Shallow Neural Network",
    "k-Nearest Neighbors (KNN)", NULL };
  GtkWidget *model_drop_down = gtk_drop_down_new_from_strings (model_names);
  gtk_drop_down_set_selected (GTK_DROP_DOWN (model_drop_down), 2);
  g_signal_connect (G_OBJECT (model_drop_down), "notify::selected-item", G_CALLBACK (model_drop_down_changed), self);
  gtk_box_append (GTK_BOX (model_vbox), model_drop_down);

  gtk_box_append (GTK_BOX (model_vbox), gtk_separator_new (GTK_ORIENTATION_HORIZONTAL));

  label = gtk_label_new ("<b>HYPERPARAMETERS</b>");
  gtk_label_set_use_markup (GTK_LABEL (label), TRUE);
  gtk_box_append (GTK_BOX (model_vbox), label);

  self->gd_vbox = gtk_box_new (GTK_ORIENTATION_VERTICAL, 0);
  gtk_box_append (GTK_BOX (self->gd_vbox), gtk_label_new ("Learning Rate α"));
  const gchar *alphas[] = { "0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30", NULL };
  self->alpha_drop_down = gtk_drop_down_new_from_strings (alphas);
  gtk_drop_down_set_selected (GTK_DROP_DOWN (self->alpha_drop_down), 4);
  gtk_box_append (GTK_BOX (self->gd_vbox), self->alpha_drop_down);
  gtk_box_append (GTK_BOX (model_vbox), self->gd_vbox);

  self->pr_vbox = gtk_box_new (GTK_ORIENTATION_VERTICAL, 0);
  gtk_box_append (GTK_BOX (self->pr_vbox), gtk_label_new ("Polynomia Degree"));
  const gchar *polynomia_degrees[] = { "0", "1", "2", NULL };
  self->polynomia_degree_drop_down = gtk_drop_down_new_from_strings (polynomia_degrees);
  gtk_drop_down_set_selected (GTK_DROP_DOWN (self->polynomia_degree_drop_down), 1);
  gtk_box_append (GTK_BOX (self->pr_vbox), self->polynomia_degree_drop_down);
  gtk_box_append (GTK_BOX (self->pr_vbox), gtk_label_new ("Regularization"));
  const gchar *regularizetion_options[] = { "None", "L1", "L2", NULL };
  self->regularization_drop_down = gtk_drop_down_new_from_strings (regularizetion_options);
  gtk_drop_down_set_selected (GTK_DROP_DOWN (self->regularization_drop_down), 0);
  gtk_box_append (GTK_BOX (self->pr_vbox), self->regularization_drop_down);
  gtk_box_append (GTK_BOX (self->pr_vbox), gtk_label_new ("Regularization Rate λ"));
  const gchar *lambdas[] = { "0", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1", "3", NULL };
  self->lambda_drop_down = gtk_drop_down_new_from_strings (lambdas);
  gtk_drop_down_set_selected (GTK_DROP_DOWN (self->lambda_drop_down), 0);
  gtk_box_append (GTK_BOX (self->pr_vbox), self->lambda_drop_down);
  gtk_box_append (GTK_BOX (model_vbox), self->pr_vbox);

  self->svm_vbox = gtk_box_new (GTK_ORIENTATION_VERTICAL, 0);
  gtk_box_append (GTK_BOX (self->svm_vbox), gtk_label_new ("Kernel"));
  const gchar *svm_kernel_names[] = { "Linear", "Radial Basis Function", NULL };
  self->svm_kernel_drop_down = gtk_drop_down_new_from_strings (svm_kernel_names);
  gtk_drop_down_set_selected (GTK_DROP_DOWN (self->svm_kernel_drop_down), 1);
  gtk_box_append (GTK_BOX (self->svm_vbox), self->svm_kernel_drop_down);
  gtk_box_append (GTK_BOX (self->svm_vbox), gtk_label_new ("Regularization Parameter C"));
  const gchar *svm_c_options[] = { "1", "3", "10", "30", "100", NULL };
  self->svm_c_drop_down = gtk_drop_down_new_from_strings (svm_c_options);
  gtk_drop_down_set_selected (GTK_DROP_DOWN (self->svm_c_drop_down), 2);
  gtk_box_append (GTK_BOX (self->svm_vbox), self->svm_c_drop_down);
  gtk_box_append (GTK_BOX (model_vbox), self->svm_vbox);

  self->knn_vbox = gtk_box_new (GTK_ORIENTATION_VERTICAL, 0);
  gtk_box_append (GTK_BOX (self->knn_vbox), gtk_label_new ("Number of Nearest Neighbors k"));
  const gchar *knn_k_options[] = { "1", "3", "10", NULL };
  self->knn_k_drop_down = gtk_drop_down_new_from_strings (knn_k_options);
  gtk_drop_down_set_selected (GTK_DROP_DOWN (self->knn_k_drop_down), 0);
  gtk_box_append (GTK_BOX (self->knn_vbox), self->knn_k_drop_down);
  gtk_box_append (GTK_BOX (model_vbox), self->knn_vbox);

  self->drawing_area = gtk_drawing_area_new ();
  gtk_widget_set_size_request (self->drawing_area, 256, 256);
  gtk_drawing_area_set_draw_func (GTK_DRAWING_AREA (self->drawing_area), draw_callback, self, NULL);

  GtkWidget *output_vbox = gtk_box_new (GTK_ORIENTATION_VERTICAL, 2);
  label = gtk_label_new ("<b>OUTPUT</b>");
  gtk_label_set_use_markup (GTK_LABEL (label), TRUE);
  gtk_box_append (GTK_BOX (output_vbox), label);
  gtk_box_append (GTK_BOX (output_vbox), self->drawing_area);
  gtk_box_append (GTK_BOX (output_vbox), gtk_separator_new (GTK_ORIENTATION_HORIZONTAL));
  label = gtk_label_new ("<b>LEARNING CURVES</b>");
  gtk_label_set_use_markup (GTK_LABEL (label), TRUE);
  gtk_box_append (GTK_BOX (output_vbox), label);

  GtkWidget *main_hbox = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 2);
  gtk_box_append (GTK_BOX (main_hbox), data_vbox);
  gtk_box_append (GTK_BOX (main_hbox), gtk_separator_new (GTK_ORIENTATION_VERTICAL));
  gtk_box_append (GTK_BOX (main_hbox), model_vbox);
  gtk_box_append (GTK_BOX (main_hbox), gtk_separator_new (GTK_ORIENTATION_VERTICAL));
  gtk_box_append (GTK_BOX (main_hbox), output_vbox);

  GtkWidget *vbox = gtk_box_new (GTK_ORIENTATION_VERTICAL, 2);
  gtk_box_append (GTK_BOX (vbox), learning_hbox);
  gtk_box_append (GTK_BOX (vbox), gtk_separator_new (GTK_ORIENTATION_HORIZONTAL));
  gtk_box_append (GTK_BOX (vbox), main_hbox);

  gtk_window_set_child (GTK_WINDOW (self), vbox);
  g_action_map_add_action_entries (G_ACTION_MAP (self), win_entries, G_N_ELEMENTS (win_entries), self);
}

GtkWidget *
le_main_window_new (GtkApplication *application)
{
  LEMainWindow *window = g_object_new (LE_TYPE_MAIN_WINDOW, "application", application, NULL);
  gtk_window_set_default_size (GTK_WINDOW (window), 256, 256);
  return GTK_WIDGET (window);
}

// cairo_surface_destroy(window->classifier_visualisation);
