from typing import List, Union

import numpy as np

from . import _ops as ops
from ._core import Tensor_from_value, gen_seq_id, gen_unique_seq_name
from ._Tensor import Tensor


def Tensor__init__(self : Tensor, shape : List[Union[int, None]], dtype = np.float32, name : str = None):
    if name is None:
        name = gen_unique_seq_name()
    self._name = name
    self._dtype = np.dtype(dtype)
    self._shape = shape

    self._seq_id = gen_seq_id()
    self._op_node = None
    self._op_node_input_tensors = []

def Tensor__str__(self : Tensor):
    if self._op_node is not None:
        return f'{self.name}({self._op_node.op_type}) {self.shape} {self.dtype}'
    else:
        return f'{self.name} {self.shape} {self.dtype}'
         
Tensor.__init__     = Tensor__init__
Tensor.__str__      = Tensor__str__
Tensor.__repr__     = Tensor__str__
Tensor.__radd__     = lambda self, value: ops.add(value, self)
Tensor.__add__      = lambda self, value: ops.add(self, value)
Tensor.__rsub__     = lambda self, value: ops.sub(value, self)
Tensor.__sub__      = lambda self, value: ops.sub(self, value)
Tensor.__rmul__     = lambda self, value: ops.mul(value, self)
Tensor.__mul__      = lambda self, value: ops.mul(self, value)
Tensor.__rtruediv__ = lambda self, value: ops.div(value, self)
Tensor.__truediv__  = lambda self, value: ops.div(self, value)
Tensor.__eq__       = lambda self, value: ops.equal(self, value)
Tensor.__or__       = lambda self, value: ops.or_(self, value)
Tensor.__ge__       = lambda self, value: ops.greater_equal(self, value)
Tensor.__gt__       = lambda self, value: ops.greater(self, value)
Tensor.__le__       = lambda self, value: ops.less_equal(self, value)
Tensor.__lt__       = lambda self, value: ops.less(self, value)
Tensor.from_value          = Tensor_from_value
Tensor.cast                = ops.cast
Tensor.equal               = ops.equal
Tensor.get_shape_as_tensor = ops.shape
Tensor.greater_equal       = ops.greater_equal
Tensor.greater             = ops.greater
Tensor.less_equal          = ops.less_equal
Tensor.less                = ops.less
Tensor.not_                = ops.not_
Tensor.or_                 = ops.or_
Tensor.reduce_sum          = ops.reduce_sum
Tensor.reshape             = ops.reshape
Tensor.squeeze             = ops.squeeze
Tensor.tile                = ops.tile
Tensor.unsqueeze           = ops.unsqueeze


