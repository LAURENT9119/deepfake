import numpy as np

from .api import OpenCL as CL

class OpenCLBuffer:
    __slots__ = ['_device','_cl_mem','_size']

    def __init__(self, device, cl_mem, size):
        OpenCLBuffer._object_count += 1
        self._device = device
        self._cl_mem = cl_mem
        self._size = size
        
    def __del__(self):
        OpenCLBuffer._object_count -= 1
        self._device._on_del_CLBuffer(self._cl_mem, self._size)
    
    def _get_cl_mem(self): return self._cl_mem
    
    def set(self, value, wait=False):
        """
        Parameters

            value   OpenCLBuffer    copy data from other OpenCLBuffer.
            
                    np.ndarray  copies values from ndarray 
                                to OpenCLBuffer's memory
            
            wait(False) bool    wait to finish operation
                                nparray can be modified while uploading the data to the device
                                thus you need to wait or not to modify nparray during whole computation cycle
        """            
        if isinstance(value, OpenCLBuffer):
            if self != value:
                if self._size != value.size:
                    raise Exception(f'Unable to copy from OpenCLBuffer with {value.size} size to buffer with {self._size} size.')

                if self._device == value.device:
                    CL.EnqueueCopyBuffer ( self._device._get_ctx_q(), value._get_cl_mem(), self._get_cl_mem(), 0,0, self._size )
                else:
                    # Transfer between devices will cause slow performance
                    self.set( value.np() )
        else:
            if not isinstance(value, np.ndarray):
                raise ValueError (f'Invalid type {value.__class__}. Must be np.ndarray.')

            if value.nbytes != self._size:
                raise ValueError(f'Value size {value.nbytes} does not match OpenCLBuffer size {self._size}.')

            # wait upload, otherwise value's memory can be disappeared
            ev = CL.ndarray_to_buffer ( self._device._get_ctx_q(), self._get_cl_mem(), value)
            if wait:
                ev.wait()

    _allowed_fill_types = (np.float16, np.float32, np.float64, np.uint8, np.int8, np.int16, np.int32, np.int64)
    def fill(self, value):
        """
        Fills buffer with scalar value.

        arguments

            value   np.float16, np.float32, np.float64, np.uint8, np.int8, np.int16, np.int32, np.int64
        """
      
        if not isinstance(value, OpenCLBuffer._allowed_fill_types ):
            raise ValueError(f'Unknown type {value.__class__}. Allowed types : {OpenCLBuffer._allowed_fill_types } ')
        
        CL.EnqueueFillBuffer (self._device._get_ctx_q(), self._get_cl_mem(), value, self._size)

    _dtype_size_dict = { np.float16 : 2,
                         np.float32 : 4,
                         np.float64 : 8,
                         np.uint8 : 1,
                         np.int8 : 1,
                         np.int16 : 2,
                         np.int32 : 4,
                         np.int64 : 8}
                         
    def np(self, shape=None, dtype=np.float32):
        """
        Returns data of buffer as np.ndarray with specified shape and dtype
        """   
        dtype_size = OpenCLBuffer._dtype_size_dict[dtype]
        
        if shape is None or len(shape) == 0:
            shape = (self._size // dtype_size,)
        out_np_value = np.empty ( shape, dtype )
        
        if out_np_value.nbytes != self._size:
            raise ValueError(f'Unable to represent OpenCLBuffer with size {self._size} as shape {shape} with dtype {dtype}')        

        CL.buffer_to_ndarray( self._device._get_ctx_q(), self._get_cl_mem(), out_np_value ).wait()
            
        return out_np_value


    def __str__(self):
        return f'OpenCLBuffer [{self._size} bytes] on {str(self._device)}'

    def __repr__(self):
        return self.__str__()

    _object_count = 0
