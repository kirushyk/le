/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LELAYER_H__
#define __LELAYER_H__

#include "letensor.h"

typedef enum LeActivation
{
    LE_ACTIVATION_NONE,
    LE_ACTIVATION_SIGMOID,
    LE_ACTIVATION_TANH,
    LE_ACTIVATION_RELU,
} LeActivation;

typedef struct LeLayer
{
    LeMatrix *weights;
    LeActivation activation;
} LeLayer;

#endif
