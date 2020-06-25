#ifndef __LEDEVICE_H__
#define __LEDEVICE_H__

#include <le/lemacros.h>

LE_BEGIN_DECLS

typedef enum LeDeviceType
{
    LE_DEVICE_TYPE_CPU,
    LE_DEVICE_TYPE_CUDA,
    LE_DEVICE_TYPE_MPS
} LeDeviceType;

LE_END_DECLS

#endif
