#ifndef __LEDEVICE_H__
#define __LEDEVICE_H__

#include <glib.h>

G_BEGIN_DECLS

typedef enum LeDeviceType
{
    LE_DEVICE_TYPE_CPU,
    LE_DEVICE_TYPE_CUDA,
    LE_DEVICE_TYPE_METAL
} LeDeviceType;

G_END_DECLS

#endif
