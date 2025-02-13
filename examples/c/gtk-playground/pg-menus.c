/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "pg-menus.h"

GMenuModel *
le_app_menu_new ()
{
  GMenu     *menu    = g_menu_new ();
  GMenu     *section = g_menu_new ();
  GMenuItem *item    = g_menu_item_new ("Quit", "app.quit");
  g_menu_append_item (section, item);
  g_menu_append_section (menu, NULL, G_MENU_MODEL (section));
  g_object_unref (section);

  return G_MENU_MODEL (menu);
}

GMenuModel *
le_menubar_new ()
{
  GMenu *menu;
  GMenu *submenu;
  GMenu *section;
  GMenu *subsection;
  menu       = g_menu_new ();
  submenu    = g_menu_new ();
  section    = g_menu_new ();
  subsection = g_menu_new ();
  g_menu_append (subsection, "Random", "win.gen::rand");
  g_menu_append (subsection, "Linearly Separable", "win.gen::linsep");
  g_menu_append (subsection, "Nested Circles", "win.gen::nested");
  g_menu_append (subsection, "SV Border", "win.gen::svb");
  g_menu_append (subsection, "Spiral", "win.gen::spiral");
  g_menu_append_submenu (section, "Generate", G_MENU_MODEL (subsection));
  g_object_unref (subsection);
  g_menu_append_section (submenu, NULL, G_MENU_MODEL (section));
  g_object_unref (section);

  section = g_menu_new ();
  g_menu_append (section, "Close", "win.close");
  g_menu_append_section (submenu, NULL, G_MENU_MODEL (section));
  g_object_unref (section);

#ifndef __APPLE__
  section = g_menu_new ();
  g_menu_append (section, "Quit", "app.quit");
  g_menu_append_section (submenu, NULL, G_MENU_MODEL (section));
  g_object_unref (section);
#endif

  g_menu_append_submenu (menu, "File", G_MENU_MODEL (submenu));
  g_object_unref (submenu);

  submenu = g_menu_new ();

  section = g_menu_new ();
  g_menu_append (section, "Dark", "win.style::dark");
  g_menu_append (section, "Light", "win.style::light");
  g_menu_append_section (submenu, NULL, G_MENU_MODEL (section));
  g_object_unref (section);

  section = g_menu_new ();
  g_menu_append (section, "SVR", "win.show_1");
  g_menu_append (section, "LR", "win.show_2");
  g_menu_append_section (submenu, NULL, G_MENU_MODEL (section));
  g_object_unref (section);

  g_menu_append_submenu (menu, "View", G_MENU_MODEL (submenu));
  g_object_unref (submenu);

  return G_MENU_MODEL (menu);
}
