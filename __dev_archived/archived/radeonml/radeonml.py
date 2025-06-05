import ctypes
from pathlib import Path
from ctypes import (POINTER, WINFUNCTYPE, byref, c_byte, c_int64,
                    c_longlong, c_size_t, c_ubyte, c_uint, c_uint32, c_ulong,
                    c_void_p, c_wchar, c_wchar_p)

RadeonML_path = str(Path(__file__).parent / 'lib' / 'windows' / 'DirectML' / 'RadeonML.dll')

dlls_by_name = {}

def dll_import(dll_name, dll_path=None):
    dll = dlls_by_name.get(dll_name, None)
    if dll is None:
        try:
            if dll_path is not None:
                dll = ctypes.cdll.LoadLibrary(dll_path)
            else:
                dll = ctypes.cdll.LoadLibrary(ctypes.util.find_library(dll_name))
        except:
            pass
        if dll is None:
            raise RuntimeError(f'Unable to load {dll_name} library.')
        dlls_by_name[dll_name] = dll

    def decorator(func):
        dll_func = getattr(dll, func.__name__)
        anno = list(func.__annotations__.values())
        dll_func.argtypes = anno[:-1]
        dll_func.restype = anno[-1]
        def wrapper(*args):
            return dll_func(*args)
        return wrapper
    return decorator


class rml_status(c_uint):
    RML_OK = 0                        # Operation is successful.
    RML_ERROR_BAD_MODEL = -100        # A model file has errors.
    RML_ERROR_BAD_PARAMETER = -110    # A parameter is incorrect.
    RML_ERROR_DEVICE_NOT_FOUND = -120 # A device was not found.
    RML_ERROR_FILE_NOT_FOUND = -130   # A model file does not exist.
    RML_ERROR_INTERNAL = -140         # An internal library error.
    RML_ERROR_MODEL_NOT_READY = -150  # A model is not ready for an operation.
    RML_ERROR_NOT_IMPLEMENTED = -160  # Functionality is not implemented yet.
    RML_ERROR_OUT_OF_MEMORY = -170    # Memory allocation is failed.
    RML_ERROR_UNSUPPORTED_DATA = -180 # An unsupported scenario.

class rml_context_params(ctypes.Structure):
    """
        device_idx(0)  uint
    
            Device index, corresponding to the backend device query result.
            Enumeration is started with 1. Use RML_DEVICE_IDX_UNSPECIFIED (0)
            for auto device selection.
    """
    _fields_ = [('device_idx', c_uint32),
               ]

    
    def __init__(self, device_idx=0):
        super().__init__()
        self.device_idx : c_uint32 = c_uint32(device_idx)
        

class rml_context(c_void_p):
    ...
    
class rml_graph(c_void_p):
    ...
    
@dll_import(RadeonML_path)
def rmlCreateDefaultContext(params : POINTER(rml_context_params), out_context : POINTER(rml_context)) -> rml_status : ...

@dll_import(RadeonML_path)
def rmlReleaseContext( context : rml_context ) -> None : ...

@dll_import(RadeonML_path)
def rmlLoadGraphFromFile(path : c_wchar_p, out_graph : POINTER(rml_graph) ) -> rml_status: ...


# params = rml_context_params(0)

# context = rml_context()
# st = rmlCreateDefaultContext(params, context)

# #rmlReleaseContext(context)

# graph = rml_graph()

# st = rmlLoadGraphFromFile(r'D:\DevelopPPP\projects\DeepFaceLive\github_project\xlib\onnxruntime\CenterFace\CenterFace.onnx', graph)
# #st = rmlLoadGraphFromFile(r'F:\DeepFaceLabCUDA9.2SSE\_internal\new_AMP_.pb', graph)

# import code
# code.interact(local=dict(globals(), **locals()))
