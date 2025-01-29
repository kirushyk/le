#ifndef __BACKENDS_CPU_LECPU_H__
#define __BACKENDS_CPU_LECPU_H__

#include <glib.h>
#include <glib-object.h>

G_BEGIN_DECLS

G_DECLARE_FINAL_TYPE (LeCpuBackend, le_cpu_backend, LE, CPU_BACKEND, GInitiallyUnowned);

LeCpuBackend * le_cpu_backend_get_instance (void);

G_END_DECLS

#endif
