import copy
from typing import List

from .api import OpenCL as CL


class OpenCLDeviceInfo:
    """
    Represents picklable OpenCL device info
    """

    def __init__(self, index=None, name=None, total_memory=None):
        self._index : int = index
        self._name : str = name
        self._total_memory : int = total_memory

    def __getstate__(self): 
        return self.__dict__.copy()

    def __setstate__(self, d):
        self.__init__()
        self.__dict__.update(d)
        
    def get_index(self) -> int:
        return self._index

    def get_name(self) -> str:
        return self._name

    def get_total_memory(self) -> int:
        return self._total_memory

    def __eq__(self, other):
        if self is not None and other is not None and isinstance(self, OpenCLDeviceInfo) and isinstance(other, OpenCLDeviceInfo):
            return self._index == other._index
        return False

    def __hash__(self):
        return self._index

    def __str__(self):
        return f"[{self._index}] {self._name} [{(self._total_memory / 1024**3) :.3}Gb]"

    def __repr__(self):
        return f'{self.__class__.__name__} object: ' + self.__str__()


class OpenCLDevicesInfo:
    """
    picklable list of OpenCLDeviceInfo
    """

    def __init__(self, devices : List[OpenCLDeviceInfo] = None):
        if devices is None:
            devices = []
        self._devices = devices
        
    def __getstate__(self): 
        return self.__dict__.copy()

    def __setstate__(self, d):
        self.__init__()
        self.__dict__.update(d)
        
    def add(self, device_or_devices : OpenCLDeviceInfo):
        if isinstance(device_or_devices, OpenCLDeviceInfo):
            if device_or_devices not in self._devices:
                self._devices.append(device_or_devices)
        elif isinstance(device_or_devices, OpenCLDevicesInfo):
            for device in device_or_devices:
                self.add(device)
    
    def copy(self):
        return copy.deepcopy(self)
            
    def get_count(self): return len(self._devices)

    def __len__(self):
        return len(self._devices)

    def __getitem__(self, key):
        result = self._devices[key]
        if isinstance(key, slice):
            return self.__class__(result)
        return result

    def __iter__(self):
        for device in self._devices:
            yield device
            
    def __str__(self):  return f'{self.__class__.__name__}:[' + ', '.join([ device.__str__() for device in self._devices ]) + ']'
    def __repr__(self): return f'{self.__class__.__name__}:[' + ', '.join([ device.__repr__() for device in self._devices ]) + ']'

_opencl_devices = None

def get_available_devices_info() -> OpenCLDevicesInfo:
    """
    returns a list of available OpenCLDeviceInfo's in a OpenCLDevicesInfo list.
    """
    global _opencl_devices
    if _opencl_devices is None:
        devices = []

        for index, device in enumerate(CL.get_devices()):
            devices.append ( OpenCLDeviceInfo(index=index, name=device.get_name(), total_memory=device.get_global_mem_size() ) )

        _opencl_devices = OpenCLDevicesInfo(devices)

    return _opencl_devices

