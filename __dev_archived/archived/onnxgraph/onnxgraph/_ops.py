from typing import Iterable
import onnx
import numpy as np

from ._core import (create_op_evaluated, Tensor_from_value, make_loop_body,
                    create_op, Tensors_from_values, get_opset_version,
                    vip_to_shape_dtype)
from ._Tensor import Tensor
from ._types import _np_to_data_type


def add(v1, v2) -> Tensor:
    return create_op_evaluated('Add', Tensors_from_values(v1,v2) )
    
def mul(v1, v2) -> Tensor:
    return create_op_evaluated('Mul', Tensors_from_values(v1,v2))
    
def div(v1, v2) -> Tensor:
    return create_op_evaluated('Div', Tensors_from_values(v1,v2))
    
def sub(v1, v2) -> Tensor:
    return create_op_evaluated('Sub', Tensors_from_values(v1,v2))

def binary_erode(input_t : Tensor, struct, iterations=1):
    """
    binary_erode
    
        input_t     Tensor[NCHW]float of 0 or 1 elements

        struct      np|Tensor[HW] structuring element
                    any dtype of 0 or 1 elements
                    
        iterations  int or Tensor[]int64
        
    applies erode of struct for every N and C separatelly
    
    C must be known dim
    """
    if input_t.dtype.type not in [np.float16, np.float32, np.float64]:
        raise ValueError('input_t.dtype.type not in [np.float16, np.float32, np.float64]')
    
    if input_t.ndim != 4:
        raise ValueError(f'input_t.ndim must == 4 (NCHW)')

    IC = input_t.shape[1]
    if IC <= 0:
        raise ValueError(f'input_t channels must be > 0')
       
    struct_t = Tensor_from_value(struct, np.float16)
    if struct_t.ndim != 2:
        raise ValueError('struct must have ndim==2')   
        
    struct_sum = reduce_sum(struct_t).cast(np.int32)
    struct_t = struct_t.unsqueeze( (0,1) )
    struct_t = tile(struct_t, (IC,1,1,1)).cast(np.float16)
    
    iterations = Tensor.from_value(iterations, np.int64)
    
    # Create loop graph
    sub_inp_iter    = Tensor([], np.int64)
    sub_inp_running = Tensor([], np.bool_)
    
    sub_inp_x          = Tensor(input_t.shape, np.float16)
    sub_inp_struct     = Tensor(struct_t.shape, np.float16)
    sub_inp_struct_sum = Tensor(struct_sum.shape, np.int32)
    
    sub_out_x = sub_inp_x    
    sub_out_x = conv2d(sub_out_x, sub_inp_struct, group=IC)    
    sub_out_x = greater_equal(sub_out_x.cast(np.int32), sub_inp_struct_sum)
    sub_out_x = sub_out_x.cast(np.float16)
    
    loop_body = make_loop_body(sub_inp_iter, sub_inp_running, [sub_inp_x, sub_inp_struct, sub_inp_struct_sum],
                               sub_inp_running, [sub_out_x, sub_inp_struct, sub_inp_struct_sum]     )
    #
    
    x = loop(loop_body, iterations, [input_t.cast(np.float16), struct_t, struct_sum])[0]
    x = x.cast(input_t.dtype)
    
    return x
    
def binary_dilate(input_t : Tensor, struct):
    """
    binary_dilate
    
        input_t     NCHW tensor  
                    float tensor of 0 or 1 elements

        struct      HW structuring element value/tensor 
                    float of 0 or 1 elements
        
    applies erode of struct for every N and C separatelly
    
    C must be known dim
    """
    if input_t.dtype.type not in [np.float16, np.float32, np.float64]:
        raise ValueError('input_t.dtype.type not in [np.float16, np.float32, np.float64]')
    
    dtype = input_t.dtype
    
    struct_t = Tensor_from_value(struct, dtype)
    
    if input_t.ndim != 4:
        raise ValueError(f'input_t.ndim must == 4 (NCHW)')

    IC = input_t.shape[1]
    if IC <= 0:
        raise ValueError(f'input_t channels must be > 0')
    if struct_t.ndim != 2:
        raise ValueError('struct must have ndim==2')   
        
    struct_t = struct_t.unsqueeze(0,1)
    struct_t = tile(struct_t, (IC,1,1,1))
    
    output_t = conv2d(input_t, struct_t, group=IC)
    
    output_t = greater(output_t.cast(np.int32), 0)
    output_t = output_t.cast(dtype)
    return output_t

def cast(t : Tensor, dtype : np.dtype):
    return create_op_evaluated('Cast', [t], to=_np_to_data_type[dtype] )

def conv2d(input_t : Tensor, kernel_t : Tensor, auto_pad='SAME_LOWER', strides=[1,1], group=1):
    """
        input_t     NCHW
        kernel_t    OIHW
    """
    return create_op_evaluated('Conv', [input_t, kernel_t], auto_pad=auto_pad, strides=strides, group=group )
    
def equal(v1, v2):
    v1,v2 = Tensors_from_values(v1,v2)
    
    ver = get_opset_version()
    if ver < 7:
        raise Exception('Equal is supported from 7+ version.')
    
    if ver < 11:
        allowed_t = [np.bool, np.int32, np.int64]
        if v1.dtype.type not in allowed_t or \
           v2.dtype.type not in allowed_t:
            raise Exception(f'Equal<11 supports only {allowed_t}')
            
            
    return create_op_evaluated('Equal', Tensors_from_values(v1,v2) )

def gather(t, indices, axis=0):
    indices = Tensor_from_value(indices, np.int64)
    
    return create_op_evaluated('Gather', [t, indices], axis=axis )
    
    
def greater_equal(v1, v2):
    if get_opset_version() < 12:
        v1,v2 = Tensors_from_values(v1,v2)
        return or_( greater(v1,v2), equal(v1, v2) )
    return create_op_evaluated('GreaterOrEqual', Tensors_from_values(v1,v2) )
    
def greater(v1, v2):
    v1,v2 = Tensors_from_values(v1,v2)
    
    ver = get_opset_version()
    if ver < 7:
        raise Exception('Greater is supported from 7+ version.')
    
    if ver < 9:
        allowed_t = [np.float16, np.float32, np.float64]
        if v1.dtype.type not in allowed_t or \
           v2.dtype.type not in allowed_t:
            raise Exception(f'Greater<9 supports only {allowed_t}')
            
    return create_op_evaluated('Greater', [v1,v2] )

def less_equal(v1, v2):
    if get_opset_version() < 12:
        raise Exception('LessOrEqual is supported from 12+ version.')
    return create_op_evaluated('LessOrEqual', Tensors_from_values(v1,v2) )
    
def less(v1, v2):
    if get_opset_version() < 9:
        raise Exception('Less is supported from 9+ version.')
    return create_op_evaluated('Less', Tensors_from_values(v1,v2) )

def loop(loop_body : onnx.GraphProto, iters_count_or_tensor, input_tensors):
    """
    creates a loop from a graph created by make_loop_body()
    
     loop_body      graph
     
     iters_count_or_tensor      number of loops via int value or Tensor int value
     
    returns Tensors list that output from graph.
    """
    input_len = len(loop_body.input)
    if input_len <= 2:
        raise ValueError('Invalid loop_body. Input count must be > 2')
    output_len = len(loop_body.output)
    if output_len <= 1:
        raise ValueError('Invalid loop_body. Output count must be > 1')
    
    input_shapes_dtypes  = [ vip_to_shape_dtype(loop_body.input[i])  for i in range(input_len)  ]
    output_shapes_dtypes = [ vip_to_shape_dtype(loop_body.output[i]) for i in range(output_len) ]
    
    shape, dtype = input_shapes_dtypes[0]
    if shape != [] or dtype != np.int64:
        raise ValueError('Invalid loop_body. input[0] must be np.int64 with shape []')
    shape, dtype = input_shapes_dtypes[1]
    if shape != [] or dtype != np.bool_:
        raise ValueError('Invalid loop_body. input[1] must be np.bool_ with shape []')
    shape, dtype = output_shapes_dtypes[0]
    if shape != [] or dtype != np.bool_:
        raise ValueError('Invalid loop_body. Output[0] must be np.bool_ with shape []')
        
    if len(input_tensors) != input_len-2:
        raise ValueError(f'len(input_tensors) != inputs of the graph {input_len-2}')
    
    if not all([ (input_t.shape == shape and input_t.dtype == dtype)  for input_t, (shape,dtype) in zip(input_tensors, input_shapes_dtypes[2:]) ]):
        raise ValueError(f'all input tensors shapes and dtypes must match graph inputs: \n{input_shapes_dtypes[2:]}')
        
    iters_count_or_tensor = Tensor.from_value(iters_count_or_tensor, dtype=np.int64)
    output_tensors = [ Tensor(x[0], x[1]) for x in output_shapes_dtypes[1:] ]
        
    create_op('Loop', [iters_count_or_tensor,Tensor.from_value(True, np.bool_)]+input_tensors, 
                            output_tensors,
                            body=loop_body )
    
    return output_tensors
    
def not_(v1):
    return create_op_evaluated('Not', [v1] )
   
def or_(v1, v2):
    if get_opset_version() < 7:
        raise Exception('Or is supported from 7+ version.')
    
    return create_op_evaluated('Or', Tensors_from_values(v1,v2) )

def arange(start, limit, delta, dtype) -> Tensor:
    ver = get_opset_version()
    if ver < 11:
        raise Exception('Range is supported from 11+ version.')
        
    start = Tensor_from_value(start, dtype)
    limit = Tensor_from_value(limit, dtype)
    delta = Tensor_from_value(delta, dtype)
    return create_op_evaluated('Range', [start, limit, delta])
    
def reduce_sum(t : Tensor, axes=None, keepdims=False) -> Tensor:
    return create_op_evaluated('ReduceSum', [t], axes=axes, keepdims=1 if keepdims else 0)
    
def reshape(t : Tensor, shape) -> Tensor:
    return create_op_evaluated('Reshape', [t, Tensor_from_value(shape, np.int64) ])
    
def shape(t):
    """
    get shape of Tensor as Tensor
    """
    return create_op_evaluated('Shape', [t] )

def squeeze(t : Tensor, axes) -> Tensor:        
    ver = get_opset_version()
    if ver >= 13:
        axes = Tensor_from_value(axes, np.int64)    
            
        return create_op_evaluated('Squeeze', [t,axes])
    else:
        #1, 11
        if not isinstance(axes, Iterable):
            raise ValueError('only Iterable of ints is supported for Squeeze<13')
        
        return create_op_evaluated('Squeeze', [t], axes=axes)
        

def tile(t : Tensor, axes) -> Tensor:
    return create_op_evaluated('Tile', [t, Tensor_from_value(axes, np.int64) ])
    
def pad(t : Tensor, pads, constant_value=None) -> Tensor:
    ver = get_opset_version()
    if ver >= 11:
        pads = Tensor_from_value(pads, np.int64)
        args = [t, pads]
        
        if constant_value is not None:
            constant_value = Tensor_from_value(constant_value, t.dtype)
            args.append(constant_value)
        
        return create_op_evaluated('Pad', args)
    else:
        raise NotImplementedError()
    
def slice(t : Tensor, starts, ends, axes=None, steps=None) -> Tensor:
    ver = get_opset_version()
    if ver >= 10:
        starts = Tensor_from_value(starts, np.int64)
        ends = Tensor_from_value(ends, np.int64)
        args = [t, starts, ends]
        
        if axes is not None:
            axes = Tensor_from_value(axes, np.int64)
            args.append(axes)
            
        if steps is not None:
            steps = Tensor_from_value(steps, np.int64)
            args.append(steps)
        
        return create_op_evaluated('Slice', args)
    else:
        raise NotImplementedError()
    
def unsqueeze(t : Tensor, axes) -> Tensor:
    ver = get_opset_version()
    if ver >= 13:            
        return create_op_evaluated('Unsqueeze', [t, Tensor_from_value(axes, np.int64) ])
    else:
        #1, 11
        if not isinstance(axes, Iterable):
            raise ValueError('only Iterable of ints is supported for Unsqueeze<13')
        
        return create_op_evaluated('Unsqueeze', [t], axes=axes)
        
