/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "pg-menus.h"

GMenuModel *
le_app_menu_new()
{
    GMenu *menu, *section;
    GMenuItem *item;
    
    menu = g_menu_new();
    
    section = g_menu_new();
    item = g_menu_item_new("Quit", "app.quit");
    g_menu_append_item(section, item);
    g_menu_append_section(menu, NULL, G_MENU_MODEL(section));
    g_object_unref(section);
    
    return G_MENU_MODEL(menu);
}

GMenuModel *
le_menubar_new()
{    
    return G_MENU_MODEL(g_menu_new());
}