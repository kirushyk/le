#ifndef __LE_MACROS_H__
#define __LE_MACROS_H__

#ifdef __cplusplus
#   define LE_BEGIN_DECLS extern "C" {
#   define LE_END_DECLS }
#else
#   define LE_BEGIN_DECLS
#   define LE_END_DECLS
#endif

#ifndef M_PI
#    define M_PI 3.14159265358979323846264338327950288
#endif

#endif
