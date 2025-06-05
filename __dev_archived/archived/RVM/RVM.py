from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from xlib.image import ImageProcessor
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class RVM:
    """
    Robust High-Resolution Video Matting with Temporal Guidance.
    from https://github.com/PeterL1n/RVM

    arguments

     device_info    ORTDeviceInfo

        use RVM.get_available_devices()
        to determine a list of avaliable devices accepted by model

    raises
     Exception
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info : ORTDeviceInfo):
        if device_info not in RVM.get_available_devices():
            raise Exception(f'device_info {device_info} is not in available devices for RVM')

        path = Path(__file__).parent / 'rvm_resnet50_fp32.onnx'
        self._device = 'cpu' if device_info.is_cpu() else 'cuda'
        self._sess = sess = InferenceSession_with_device(str(path), device_info)
        
        self._reset_recurrent_state()

        self._HW = (0,0)

    def _reset_recurrent_state(self):
        self._io = io = self._sess.io_binding()
        # Set output binding.
        for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
            io.bind_output(name, self._device)
        self._rec = [ ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=np.float32), self._device) ] * 4


    def extract(self, img, reset_recurrent_state=False):
        """
        arguments

         img    np.ndarray      HW,HWC,NHWC uint8/float32

        returns (N,H,W) alpha matte [0 .. 1.0]
        """
        ip = ImageProcessor(img)
        _,H,W,_ = ip.get_dims()

        if self._HW != (H,W) or reset_recurrent_state:
            # HW is changed, reset recurrent state
            self._HW = (H,W)
            self._reset_recurrent_state()

        src = ip.to_ufloat32().ch(3).swap_ch().get_image('NCHW')

        io = self._io
        io.bind_cpu_input('src', src)
        io.bind_ortvalue_input('r1i', self._rec[0])
        io.bind_ortvalue_input('r2i', self._rec[1])
        io.bind_ortvalue_input('r3i', self._rec[2])
        io.bind_ortvalue_input('r4i', self._rec[3])
        io.bind_cpu_input('downsample_ratio', np.float32([1.0]) )

        self._sess.run_with_iobinding(io)

        _, pha, *self._rec = io.get_outputs()
        pha = pha.numpy()[:,0,:,:]
        return pha


