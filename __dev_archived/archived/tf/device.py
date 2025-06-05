import copy
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import List
import platform

class TFDeviceInfo:
    """
    Represents picklable TF device info
    """

    def __init__(self, index=None, type=None, name=None, memory=None):
        if type is not None and type not in ['CPU', 'GPU', 'DML']:
            raise ValueError("Type must be ['CPU', 'GPU', 'DML']")
        
        self._index : int = index
        self._type : str = type
        self._name : str = name
        self._memory : int = memory
    
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, d):
        self.__init__()
        self.__dict__.update(d)
        
    def is_cpu(self) -> bool: return self._type == 'CPU'
    def is_gpu(self) -> bool: return self._type == 'GPU'
    def is_dml(self) -> bool: return self._type == 'DML'
        
    def get_index(self) -> int: return self._index
    def set_index(self, index): self._index = index
        
    def get_type(self) -> str:
        return self._type
        
    def get_name(self) -> str:
        return self._name
        
    def get_tf_device_name(self):
        return f'/{self._type}:{self._index}'
        
    def get_backend_name(self) -> str:
        return 'tf'
        
    def get_total_memory(self) -> int:
        return self._memory
        
    def get_free_memory(self) -> int:
        return self._memory
    
    def __eq__(self, other):
        if self is not None and other is not None and isinstance(self, TFDeviceInfo) and isinstance(other, TFDeviceInfo):
            return self._index == other._index and self._type == other._type
        return False

    def __hash__(self):
        return self._index

    def __str__(self):
        if self.is_cpu():
            return f"CPU"
        else:
            return f"[{self._index}] {self._name} [{(self._memory / 1024**3) :.3}Gb]"
    
    def __repr__(self):
        return f'{self.__class__.__name__} object: ' + self.__str__()


class TFDevicesInfo:
    """
    picklable list of TFDeviceInfo
    """
    def __init__(self, devices : List[TFDeviceInfo] = None):
        if devices is None:
            devices = []
        self._devices = devices
        
    def __getstate__(self): 
        return self.__dict__.copy()

    def __setstate__(self, d):
        self.__init__()
        self.__dict__.update(d)
        
    def add(self, device_or_devices : TFDeviceInfo):
        if isinstance(device_or_devices, TFDeviceInfo):
            if device_or_devices not in self._devices:
                self._devices.append(device_or_devices)
        elif isinstance(device_or_devices, TFDevicesInfo):
            for device in device_or_devices:
                self.add(device)
    
    def copy(self):
        return copy.deepcopy(self)
            
    def get_count(self): return len(self._devices)
    
    def get_largest_total_memory_device(self) -> TFDeviceInfo:
        raise NotImplementedError()
        
    def get_smallest_total_memory_device(self) -> TFDeviceInfo:
        raise NotImplementedError()

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
           


_tf_devices_info = None

def get_cpu_device() -> TFDeviceInfo:
    return TFDeviceInfo(index=0, type='CPU', name='CPU', memory=0)

def get_available_devices_info(include_cpu=True, cpu_only=False) -> TFDevicesInfo:
    """
    returns a list of available TFDeviceInfo's in a DevicesInfo list.
    
    """
    global _tf_devices_info
    if _tf_devices_info is None:
        _initialize_tf_devices()
        devices = []
        if not cpu_only:
            for i in range ( int(os.environ['TF_DEVICES_COUNT']) ):
                devices.append ( TFDeviceInfo(index=i,
                                            type=os.environ[f'TF_DEVICE_{i}_TYPE'],
                                            name=os.environ[f'TF_DEVICE_{i}_NAME'],
                                            memory=int(os.environ[f'TF_DEVICE_{i}_MEM'])) )
        if include_cpu or cpu_only:
            devices.append ( get_cpu_device() )
        _tf_devices_info = TFDevicesInfo(devices)

    return _tf_devices_info
    
def _get_tf_devices_proc(q : multiprocessing.Queue):
    if platform.system() == 'Windows':
        compute_cache_path = Path(os.environ['APPDATA']) / 'NVIDIA' / ('CACHE_ALL')
        os.environ['CUDA_CACHE_PATH'] = str(compute_cache_path)
        if not compute_cache_path.exists():
            #print("Caching GPU kernels...")
            compute_cache_path.mkdir(parents=True, exist_ok=True)
            
    os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # tf log errors only
        
    try:
        import tensorflow
    except:
        q.put([])
        time.sleep(0.1)
        return
    
    tf_version = tensorflow.version.VERSION
    if tf_version[0] == 'v':
        tf_version = tf_version[1:]
    if tf_version[0] == '2':
        tf = tensorflow.compat.v1
    else:
        tf = tensorflow
    
    import logging

    # Disable tensorflow warnings
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.setLevel(logging.ERROR)

    from tensorflow.python.client import device_lib

    devices = []
    
    physical_devices = device_lib.list_local_devices()
    physical_devices_f = {}
    for dev in physical_devices:
        dev_type = dev.device_type
        dev_tf_name = dev.name
        dev_tf_name = dev_tf_name[ dev_tf_name.index(dev_type) : ]
        
        dev_idx = int(dev_tf_name.split(':')[-1])
        
        if dev_type in ['GPU','DML']:
            dev_name = dev_tf_name
            
            dev_desc = dev.physical_device_desc
            if len(dev_desc) != 0:
                if dev_desc[0] == '{':
                    dev_desc_json = json.loads(dev_desc)
                    dev_desc_json_name = dev_desc_json.get('name',None)
                    if dev_desc_json_name is not None:
                        dev_name = dev_desc_json_name
                else:
                    for param, value in ( v.split(':') for v in dev_desc.split(',') ):
                        param = param.strip()
                        value = value.strip()
                        if param == 'name':
                            dev_name = value
                            break
            
            physical_devices_f[dev_idx] = (dev_type, dev_name, dev.memory_limit)
                    
    q.put(physical_devices_f)
    time.sleep(0.1)
    

def _initialize_tf_devices():
    """
    Determine available Tensorflow devices, and place info about them to os.environ,
    they will be available in spawned subprocesses.
    """
    if int(os.environ.get('TF_DEVICES_INITIALIZED', 0)) == 0:
        
        os.environ['TF_DEVICES_INITIALIZED'] = '1'
        os.environ['TF_DEVICES_COUNT'] = '0'
        
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            os.environ.pop('CUDA_VISIBLE_DEVICES')
        
        os.environ['CUDA_​CACHE_​MAXSIZE'] = '2147483647'
        
        multiprocessing.set_start_method('spawn', force=True)
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_get_tf_devices_proc, args=(q,), daemon=True)
        p.start()
        p.join()
        devices = q.get()
        
        os.environ['TF_DEVICES_COUNT'] = str(len(devices))
        for i in devices:
            dev_type, name, mem = devices[i]            
            
            os.environ[f'TF_DEVICE_{i}_TYPE'] = dev_type
            os.environ[f'TF_DEVICE_{i}_NAME'] = name
            os.environ[f'TF_DEVICE_{i}_MEM'] = str(mem)

_initialize_tf_devices()
