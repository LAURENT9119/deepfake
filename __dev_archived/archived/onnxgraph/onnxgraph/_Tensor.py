from typing import List, Union

import numpy as np


class Tensor:

    def __init__(self, shape : List[Union[int, None]], dtype = np.float32, name : str = None):
        """
        represents onnx Tensor

        arguments

            shape               List[Union[int, None]]

            dtype(np.float32)   numpy dtype

            name(None)          should be used for input tensors
        """

    @staticmethod
    def from_value(value, dtype=None) -> 'Tensor':
        """
        create a Constant tensor from value

        arguments

         value      Tensor  , - do nothing, returns the same
                    list of int/float , dtype must be specified
                    single int, float, numpy.dtype value , dtype must be specified
                    np.ndarray

         dtype(None)   np.dtype
        """
        ...

    # Properties are non-obvious shit, but these are minimal to match numpy behaviour
    @property
    def shape(self) -> List[Union[int, None]]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        else:
            return f'#{self._seq_id}'

    def __radd__(self, value) -> 'Tensor':      ...
    def __add__(self, value)  -> 'Tensor':      ...
    def __rsub__(self, value) -> 'Tensor':      ...
    def __sub__(self, value)  -> 'Tensor':      ...
    def __rmul__(self, value) -> 'Tensor':      ...
    def __mul__(self, value)  -> 'Tensor':      ...
    def __rtruediv__(self, value) -> 'Tensor':  ...
    def __truediv__(self, value)  -> 'Tensor':  ...
    def __eq__(self, value)  -> 'Tensor':  ...
    def __ge__(self, value)  -> 'Tensor':  ...
    def __gt__(self, value)  -> 'Tensor':  ...
    def __le__(self, value)  -> 'Tensor':  ...
    def __lt__(self, value)  -> 'Tensor':  ...

    def __str__(self):  ...
    def __repr__(self): ...

    def cast(self, dtype) -> 'Tensor':
        """
        """

    def equal(self, v2) -> 'Tensor':
        """
        """

    def get_shape_as_tensor(self) -> 'Tensor':
        """
        get shape as tensor
        """

    def greater_equal(self, v2) -> 'Tensor':
        """
        """

    def greater(self, v2) -> 'Tensor':
        """
        """

    def less_equal(self, v2) -> 'Tensor':
        """
        """

    def less(self, v2) -> 'Tensor':
        """
        """

    def not_(self) -> 'Tensor':
        """
        """

    def or_(self, v2) -> 'Tensor':
        """
        """

    def reduce_sum(self, axes) -> 'Tensor':
        """
        """

    def reshape(self, shape) -> 'Tensor':
        """
        """

    def squeeze(self, axes) -> 'Tensor':
        """
        """

    def tile(self, axes) -> 'Tensor':
        """
        """

    def unsqueeze(self, axes) -> 'Tensor':
        """
        """



