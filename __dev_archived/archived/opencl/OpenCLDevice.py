import numpy as np
from .device import OpenCLDeviceInfo

from .api import OpenCL as CL
from .OpenCLBuffer import OpenCLBuffer

class OpenCLDevice:
    """
    Represents physical OpenCL device
    """

    def __init__(self, device):
        self._device = device
        
        self._name = device.get_name()
        self._global_mem_size = device.get_global_mem_size()
        

        self._cached_kernels = {} # Cached kernels
        self._ctx_q = None
        self._ctx = None

        self._total_memory_allocated = 0
    
    def __str__(self):
        return f"{self._name} [{(self._global_mem_size / 1024**3) :.3}Gb]"

    def __repr__(self):
        return f'{self.__class__.__name__} object: ' + self.__str__()

    def get_total_allocated_memory(self):
        return self._total_memory_allocated
        
    def get_max_work_group_size(self):
        return self._device.get_max_work_group_size()

    def alloc_for_np(self, np_ar : np.ndarray, init_func=None) -> OpenCLBuffer:
        """
        Instant alloc OpenCLBuffer to fit provided ndarray
        """
        return self.alloc(np_ar.nbytes, init_func=init_func)
    
    def alloc(self, size, init_func=None) -> OpenCLBuffer:
        """
        Instant alloc OpenCLBuffer with given size
        
        To free OpenCLBuffer just lose all references to it.
                
         size       int     size of buffer in bytes

         init_func(None)    function that receives (OpenCLBuffer) argument.
                            Called when memory is physically allocated
                            and should be initialized
                            
        returns OpenCLBuffer if success
                None if memory error
                raises Exception on other errors
        """
        
        if size <= 0:
            raise ValueError(f'alloc with {size} size.')
        
        try:
            cl_mem = CL.CreateBuffer(self._get_ctx(), size)
            # Fill one byte to check memory availability
            CL.EnqueueFillBuffer(self._get_ctx_q(), cl_mem, np.uint8(0), 1, 0)
            self._total_memory_allocated += size
            
            cl_buf = OpenCLBuffer(self, cl_mem, size)
            if init_func is not None:
                init_func(cl_buf)
            return cl_buf
        
        except CL.Exception as e:
            if e.error.value == CL.ERROR.MEM_OBJECT_ALLOCATION_FAILURE:
                return None
            else:
                raise Exception(f"Unable to allocate {size // 1024**2}Mb on {str(self)}. {e}")
        except Exception as e:
            raise Exception(f"Unable to allocate {size // 1024**2}Mb on {str(self)}. {e}")
        
        
    def run(self, kernel, *args, global_shape=None, local_shape=None, global_shape_offsets=None):
        """
        Run kernel on this Device

        Arguments

            *args           arguments will be passed to OpenCL kernel
                            allowed types:
                            OpenCLBuffer
                            np.int32 np.int64 np.uint32 np.uint64 np.float32

            global_shape(None)  tuple of ints, up to 3 dims
                                amount of parallel kernel executions.
                                in OpenCL kernel,
                                id can be obtained via get_global_id(dim)

            local_shape(None)   tuple of ints, up to 3 dims
                                specifies local groups of every dim of global_shape.
                                in OpenCL kernel,
                                id can be obtained via get_local_id(dim)

            global_shape_offsets(None)  tuple of ints
                                        offsets for global_shape


        """

        krn = self._cached_kernels.get(kernel, None)
        if krn is None:
            # Build kernel on the fly
            krn = CL.CreateProgramWithSource(self._get_ctx(), kernel.kernel_text)

            try:
                CL.BuildProgram(krn, [self._device], options="-cl-std=CL1.2 -cl-single-precision-constant")
            except CL.Exception as e:
                raise Exception(f'Build kernel fail: {CL.GetProgramBuildInfo(krn, CL.PROGRAM_BUILD_INFO.BUILD_LOG, self._device)}')

            kernels = CL.CreateKernelsInProgram(krn)
            if len(kernels) != 1:
                raise ValueError('CLKernel must contain only one __kernel.')
            krn = self._cached_kernels[kernel] = kernels[0]

        krn(self._get_ctx_q(), global_shape, local_shape, global_shape_offsets,
            *[arg._get_cl_mem() if isinstance(arg, OpenCLBuffer) else arg for arg in args])
    

    def wait(self):
        """
        Wait to finish all queued operations on this Device
        """
        CL.Finish(self._get_ctx_q())    
    
    def cleanup(self):
        """
        Frees all resources from this CLDevice.
        """
        if self._total_memory_allocated != 0:
            raise Exception('Unable to cleanup CLDevice, while not all OpenCLBuffer`s are deallocated.')

        # Lose reference is enough to free OpenCL resources.
        self._cached_kernels = {}
        self._ctx_q = None
        self._ctx = None
        
    def _on_del_CLBuffer(self, cl_mem, size):
        self._total_memory_allocated -= size
    

    def _get_ctx(self):
        # Create OpenCL context on demand
        if self._ctx is None:
            self._ctx = CL.CreateContext(devices=[self._device])
        return self._ctx

    def _get_ctx_q(self):
        # Create CommandQueue on demand
        if self._ctx_q is None:
            self._ctx_q = CL.CreateCommandQueue(self._get_ctx(), self._device)
        return self._ctx_q
        

    
_devices = None
def get_device( device_info : OpenCLDeviceInfo ) -> OpenCLDevice:
    """
    returns OpenCLDevice from OpenCLDeviceInfo
    """
    global _devices
    if _devices is None:
        _devices = [ OpenCLDevice(phys_dev) for phys_dev in CL.get_devices() ]            
    
    return _devices[device_info.get_index()]
    