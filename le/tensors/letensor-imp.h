/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LETENSOR_IMP_H__
#define __LETENSOR_IMP_H__

#include <stdbool.h>
#include "letype.h"
#include "leshape.h"
#include "../../backends/ledevice.h"
#include "../lebackend.h"

struct _LeTensor
{
  GObject             parent;
  LeType              element_type;
  LeShape            *shape;
  bool                owns_data;
  /// @note: In dimension of lowest order
  guint32             stride;
  LeDeviceType        device_type;
  LeBackendInterface *backend_interface;
  void               *data;
};

#endif

