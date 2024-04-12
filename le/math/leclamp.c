#include "leclamp.h"

gfloat
le_clamp_f32 (gfloat v, gfloat min, gfloat max)
{
  return (v > max) ? max : ((v < min) ? min : v);
}
