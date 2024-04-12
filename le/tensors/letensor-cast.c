#include "letensor-cast.h"
#include <assert.h>

#define DEFINE_SIMPLE_CAST_FN(dst_type, dsttype_name, src_type, srctype_name) \
void \
dsttype_name ## _ ## srctype_name(void *dst, void *src, gsize index) \
{ \
    ((dst_type *)dst)[index] = ((src_type *)src)[index]; \
}

DEFINE_SIMPLE_CAST_FN(gfloat, f32, guint8, u8)
DEFINE_SIMPLE_CAST_FN(guint32, u32, guint8, u8)
DEFINE_SIMPLE_CAST_FN(guint8, u8, guint32, u32)

bool le_cast_rawcpy[LE_TYPE_COUNT][LE_TYPE_COUNT] =
{
    /* to\from  void   i8     u8     i16    u16    i32    u32    f16    f32    f64 */
    /* void */ {false, false, false, false, false, false, false, false, false, false},
    /* i8   */ {false, true,  true,  false, false, false, false, false, false, false},
    /* u8   */ {false, true,  true,  false, false, false, false, false, false, false},
    /* i16  */ {false, false, false, true,  true,  false, false, false, false, false},
    /* u16  */ {false, false, false, true,  true,  false, false, false, false, false},
    /* i32  */ {false, false, false, false, false, true,  true,  false, false, false},
    /* u32  */ {false, false, false, false, false, true,  true,  false, false, false},
    /* f16  */ {false, false, false, false, false, false, false, true,  false, false},
    /* f32  */ {false, false, false, false, false, false, false, false, true,  false},
    /* f64  */ {false, false, false, false, false, false, false, false, false, true}
};

LeCast le_cast_fn[LE_TYPE_COUNT][LE_TYPE_COUNT] =
{
    /* to\from  void     i8       u8       i16      u16      i32      u32      f16      f32      f64 */
    /* void */ {NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL},
    /* i8   */ {NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL},
    /* u8   */ {NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    u8_u32,  NULL,    NULL,    NULL},
    /* i16  */ {NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL},
    /* u16  */ {NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL},
    /* i32  */ {NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL},
    /* u32  */ {NULL,    NULL,    u32_u8,  NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL},
    /* f16  */ {NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL},
    /* f32  */ {NULL,    NULL,    f32_u8,  NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL},
    /* f64  */ {NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL,    NULL}
};
