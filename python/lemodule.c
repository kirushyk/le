#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *
le_tensor(PyObject *self, PyObject *args)
{
    int sts = 0;
    return PyLong_FromLong(sts);
}

static PyMethodDef LeMethods[] =
{
    {"tensor", le_tensor, METH_VARARGS, "Create a Le Tensor."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef lemodule = 
{
    PyModuleDef_HEAD_INIT,
    "le", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module,
           or -1 if the module keeps state in global variables. */
    LeMethods
};

PyMODINIT_FUNC
PyInit_le(void)
{
    return PyModule_Create(&lemodule);
}
