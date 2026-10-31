#include <Python.h>
static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "exllamav3_ext", NULL, -1, NULL};
PyMODINIT_FUNC PyInit_exllamav3_ext(void) {
    PyObject *m = PyModule_Create(&def);
    PyObject *stub = PyImport_ImportModule("_exl3_ext_stub");
    if (!m || !stub) return NULL;
    PyObject *ga = PyObject_GetAttrString(stub, "__getattr__");
    PyModule_AddObject(m, "__getattr__", ga);
    return m;
}
