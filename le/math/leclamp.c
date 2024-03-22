#include "leclamp.h"

float
le_clamp_f32 (float v, float min, float max)
{
  return (v > max) ? max : ((v < min) ? min : v);
}
