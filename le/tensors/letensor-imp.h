/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LETENSOR_IMP_H__
#define __LETENSOR_IMP_H__

#include <stdbool.h>
#include "letype.h"
#include "leshape.h"
#include "../../platform/ledevice.h"

struct LeTensor
{
    LeType        element_type;
    LeShape      *shape;
    bool          owns_data;
    /// @note: In dimension of lowest order
    uint32_t      stride;
    LeDeviceType  device_type;
    void         *data;
};

#endif

