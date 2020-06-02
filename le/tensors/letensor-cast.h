#include <stdbool.h>
#include "letype.h"

typedef void (*LeCast)(void *dst, void *src, size_t index);

extern bool le_cast_rawcpy[LE_TYPE_COUNT][LE_TYPE_COUNT];

extern LeCast le_cast_fn[LE_TYPE_COUNT][LE_TYPE_COUNT];
