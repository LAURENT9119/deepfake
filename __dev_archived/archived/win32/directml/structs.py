from ctypes import c_uint, c_int, Structure

class DML_CREATE_DEVICE_FLAGS(c_uint):
    DML_CREATE_DEVICE_FLAG_NONE = 0
    DML_CREATE_DEVICE_FLAG_DEBUG = 0x1