import os
import sys
from pathlib import Path
from typing import List, Union

from xlib.time import timeit

from .device import TFDeviceInfo, TFDevicesInfo, get_cpu_device


class TFSession:
    
    def __init__(self, devices_info : TFDevicesInfo, data_format='NCHW' ):
        if not isinstance(devices_info, TFDevicesInfo ):
            raise ValueError(f'devices_info must be TFDevicesInfo')
        
        if len(devices_info) == 0:
            raise ValueError('devices_info must contain at least 1 device')
        
        devices_info = devices_info.copy()
        
        is_caching_GPU_kernels = False
        if sys.platform[0:3] == 'win':
            if any([x.is_gpu() for x in devices_info ]):
                # define compute cache path for GPU backend (CUDA)
                if all( [ device_info.get_name() == devices_info[0].get_name() for device_info in devices_info ] ):
                    # all names are the same
                    devices_str = '_' + devices_info[0].get_name().replace(' ','_')
                else:
                    devices_str = ""
                    for device_info in devices_info:
                        devices_str += '_' + device_info.get_name().replace(' ','_')
                        
                    compute_cache_path = Path(os.environ['APPDATA']) / 'NVIDIA' / ('CACHE' + devices_str)
                    if not compute_cache_path.exists():
                        is_caching_GPU_kernels = True
                        compute_cache_path.mkdir(parents=True, exist_ok=True)
                    os.environ['CUDA_CACHE_PATH'] = str(compute_cache_path)
        
        if is_caching_GPU_kernels:
            print("Caching GPU kernels...")
        
        os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # tf log errors only
        
        if 'tensorflow' in sys.modules:
            raise Exception('tensorflow is already imported which is not allowed to create TFSession')

        import tensorflow
        
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
        
        if tf_version[0] == '2':
            tf.disable_v2_behavior()
       
        self._tf : tensorflow = tf
        
        gpu_devices_info = [ x for x in devices_info if x.is_gpu() or x.is_dml() ]
            
        if len(gpu_devices_info) == 0:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto()
            config.gpu_options.visible_device_list = ','.join( [ str(gpu_device_info.get_index()) for gpu_device_info in gpu_devices_info ] )
            
            for i,gpu_device_info in enumerate(gpu_devices_info):
                gpu_device_info.set_index(i)
                
        config.gpu_options.force_gpu_compatible = True
        config.gpu_options.allow_growth = True
        self._tf_sess_config = config
        
        self._tf_graph = self._tf.Graph()
        self._tf_sess = None#self._tf.Session(graph=self._tf_graph, config=self._tf_sess_config) 
        self._devices_info = devices_info
        self._data_format = data_format
        
    def get_tf(self): return self._tf
    
    def get_session(self, graph=None):
        """
        get or create session with graph
        """
        if self._tf_sess is None:
            self._tf_sess = self._tf.Session(graph=graph, config=self._tf_sess_config)
        return self._tf_sess
        
    
    def get_floatx(self):
        return self._tf.float32
    
    def get_data_format(self):
        return self._data_format
        
    def get_devices_count(self): return len(self._devices_info)
    
    def get_tf_device_name(self, id : int):
        return self._devices_info[id].get_tf_device_name()

    
    def using_device(self, id : int):
        """
        
        """
        return self._tf.device( self._devices_info[id].get_tf_device_name() )

class TFInferenceSession:
    """
    Initialize singleton tensorflow model inferense session per process with specified devices_info  
    """

    def __init__(self, model_path : Union[Path, str], 
                       in_tensor_names : List[str], 
                       out_tensor_names : List[str],
                       device_info : TFDeviceInfo = None):
                       
        if device_info is None:
            device_info = get_cpu_device()
        if not isinstance(device_info, TFDeviceInfo ):
            raise ValueError(f'devices_info must be device_info')
        
        self._tf_session = TFSession( TFDevicesInfo([device_info]) )
        
        model_path = self._model_path = Path(model_path)
        if not model_path.exists():
            raise Exception(f'{model_path} does not exist.')
        if model_path.suffix != '.pb':
            raise ValueError('only .pb file is supported')
        
        tf = self._tf_session.get_tf()
        
        graph = self._graph = tf.Graph()
        self._tf_sess = self._tf_session.get_session(graph=graph)
        
        tf_device_name = self._tf_session.get_tf_device_name(0)
        print(tf_device_name)
        with tf.gfile.GFile(str(model_path), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            for node in graph_def.node:
                node.device = tf_device_name
            
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")

        self._in_tensors = [ graph.get_tensor_by_name(in_tensor_name) for in_tensor_name in in_tensor_names]
        self._out_tensors = [ graph.get_tensor_by_name(out_tensor_name) for out_tensor_name in out_tensor_names]


    def run(self, in_data : List):
        if len(in_data) != len(self._in_tensors):
            raise ValueError('len in_data must match in_tensors')

        return self._tf_sess.run (self._out_tensors, 
                                  feed_dict={self._in_tensors[i] : d for i,d in enumerate(in_data)} )
