import numpy as np
from onnx import TensorProto

_np_to_data_type = {
    np.bool_ : TensorProto.BOOL,
    np.uint8 : TensorProto.UINT8,
    np.uint16 : TensorProto.UINT16,
    np.uint32 : TensorProto.UINT32,
    np.uint64 : TensorProto.UINT64,
    np.int8 : TensorProto.INT8,
    np.int16 : TensorProto.INT16,
    np.int32 : TensorProto.INT32,
    np.int64 : TensorProto.INT64,
    np.float16 : TensorProto.FLOAT16,
    np.float32 : TensorProto.FLOAT,
    np.float64 : TensorProto.DOUBLE,
}
_data_type_to_np = { _np_to_data_type[x]:x for x in _np_to_data_type }

q1 = { np.dtype(x).name : _np_to_data_type[x] for x in _np_to_data_type }
q2 = { np.dtype(x)      : _np_to_data_type[x] for x in _np_to_data_type }
_np_to_data_type.update(q1)
_np_to_data_type.update(q2)

