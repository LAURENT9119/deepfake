import operator
import threading
from typing import Iterable, List, Tuple, Union

import numpy as np
import onnx
from onnx import NodeProto, ValueInfoProto
from onnx import helper as onnx_helper
from onnx.onnx_ml_pb2 import TensorProto

from ._Tensor import Tensor
from ._types import _data_type_to_np, _np_to_data_type

_seq_id = 0
_name_seq_id = 0
_seq_id_lock = threading.Lock()
_opset_version = 12

def get_opset_version() -> int:
    global _opset_version
    return _opset_version

def set_opset_version(version : int):
    """
    set current opset version for Tensors and models.
    Do not change during graph building.
    """
    global _opset_version
    _opset_version = version


def gen_unique_seq_name():
    global _name_seq_id
    _seq_id_lock.acquire()
    result = _name_seq_id = _name_seq_id + 1
    _seq_id_lock.release()
    return f'#{result}'

def gen_seq_id():
    global _seq_id
    _seq_id_lock.acquire()
    result = _seq_id = _seq_id + 1
    _seq_id_lock.release()
    return result

def Tensor_to_vip(t : Tensor) -> onnx.ValueInfoProto:
    return onnx_helper.make_tensor_value_info(t.name, _np_to_data_type[t.dtype], t.shape)

def vip_to_shape_dtype(vip : onnx.ValueInfoProto):
    t = vip.type.tensor_type
    return [dim.dim_value for dim in t.shape.dim], _data_type_to_np[t.elem_type]
    
def assert_tensor_shape_dtype(t : Tensor, shape, dtype):
    if shape != t.shape or t.dtype != np.dtype(dtype):
        raise ValueError(f'Tensor must be {shape} {np.dtype(dtype).name}')
        
def backward_gather(tensor_list : List['Tensor'],
                    exclude_tensor_list=None) -> Tuple[ List[NodeProto], List[ValueInfoProto] ]:
    """
    backward gather all nodes and inputs start from tensor_list

    returns (nodes_list, input_vip_list)
    """
    tensor_list = tensor_list.copy()

    processed_t_set = set(t._seq_id for t in exclude_tensor_list) if exclude_tensor_list is not None else set()  
    processed_node_set = set()
    uniq_t_name_set = set()

    attr_seq_id = operator.attrgetter('_seq_id')

    nodes = []
    input_vips = []
    while len(tensor_list) != 0:
        # Process tensor with largest _seq_id (most late)
        tensor_list = sorted(tensor_list, key=attr_seq_id)
        t = tensor_list.pop(-1)

        # Check already processed tensors
        t_seq_id = t._seq_id
        if t_seq_id in processed_t_set:
            continue
        processed_t_set.add(t_seq_id)

        # Check tensor name duplicate
        t_name = t.name
        if t_name in uniq_t_name_set:
            raise Exception(f'Duplicate tensor name {t_name}')
        uniq_t_name_set.add(t_name)

        op_node = t._op_node
        if op_node is not None:
            # Tensor is produced by op
            
            # Multiple tensors can have same producer op, for example loop.
            # thus check if already processed
            node_id = id(op_node)
            if node_id not in processed_node_set:
                # Node is not processed before. 
                processed_node_set.add(node_id)
                nodes.append(op_node)
                tensor_list.extend(t._op_node_input_tensors)
        else:
            # otherwise it is an input tensor
            # collect ValueInfoProto's
            input_vips.append( Tensor_to_vip(t) )

    return nodes[::-1], input_vips[::-1]

def Tensors_from_values(v1, v2):
    if isinstance(v1, Tensor):
        if isinstance(v2, Tensor):
            return [v1, v2]
        else:
            return [v1, Tensor_from_value(v2, dtype=v1.dtype)]
    else:
        if isinstance(v2, Tensor):
            return [Tensor_from_value(v1, dtype=v2.dtype), v2]
        else:
            raise ValueError('one of values must be a Tensor')

def Tensor_from_value(value, dtype=None) -> Tensor:
    """
    create a Constant tensor from value

    arguments

        value       Tensor  , - returns if success check of dtype if specified

                    Iterable of int/float , dtype must be specified

                    single int, float, numpy.dtype value , dtype must be specified

                    np.ndarray

        dtype(None)   np.dtype
    """
    if isinstance(value, Tensor):
        if dtype is not None:
            if value.dtype != dtype:
                raise ValueError(f'Tensor {value} must have dtype {dtype}')
        return value

    elif not isinstance(value, np.ndarray):

        if isinstance(value, Iterable):
            if dtype is None:
                raise ValueError('dtype must be specified for list of values')
            value = np.array(value, dtype)
        elif isinstance(value, (int, float, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64, np.bool_) ):
            if dtype is None:
                raise ValueError('dtype must be specified for single value')
            value = np.array(value, dtype)
        else:
            raise ValueError(f'Unsupported type of value {value.__class__}')

    if dtype is not None:
        value = value.astype(dtype)

    value = onnx_helper.make_tensor(name=gen_unique_seq_name(),
                                    data_type=_np_to_data_type[value.dtype],
                                    dims=value.shape,
                                    vals=value.view(dtype=np.uint16).flatten() if value.dtype == np.float16 else value.flatten() )

    return create_op_evaluated('Constant', [], value=value)


def create_op(op_name, input_tensors : List['Tensor'], output_tensors : List['Tensor'], **attrs) -> Tensor:
    """
    create op with known list of input/output tensors of the op
    """
    node = onnx_helper.make_node(op_name, [t.name for t in input_tensors],
                                          [t.name for t in output_tensors], **attrs)
    for output_t in output_tensors:
        if output_t._op_node is not None:
            raise ValueError(f'{output_t} already has a node.')
        output_t._op_node_input_tensors = input_tensors
        output_t._op_node = node

def create_op_evaluated(op_name, input_tensors : List['Tensor'],  **attrs) -> Tensor:
    """
    create op with evaluated single output Tensor
    """
    output_t_name = gen_unique_seq_name()

    node = onnx_helper.make_node(op_name, [t.name for t in input_tensors],
                                          [output_t_name], **attrs)

    #pre_nodes = []
    #pre_vips = []
    
    # ver = get_opset_version()
    
    # if op_name in ['Reshape','Tile','Gather'] or \
    #    (ver >= 10 and op_name in ['Slice']) or \
    #    (ver >= 11 and op_name in ['Pad']) or \
    #    (ver >= 13 and op_name in ['Squeeze','Unsqueeze']):
    #     # some ops have arg from Tensor at runtime, 
    #     # thus we need to gather all nodes from that arg in order to evaluate output shape at compile time
    #     # if Tensor shape is input by user, then there is no chance to evaluate the shape at compile time
    #     pre_nodes, pre_vips = backward_gather(input_tensors[1:], exclude_tensor_list=[input_tensors[0]])
    # elif (ver >= 11 and op_name in ['Range']):
    pre_nodes, pre_vips = backward_gather(input_tensors)
        
    # make model and eval the shape
    model = onnx_helper.make_model(onnx_helper.make_graph(pre_nodes+[node], '', pre_vips+[Tensor_to_vip(t) for t in input_tensors], []))
    model.ir_version=7
    model.opset_import[0].version=get_opset_version()

    model_out = onnx.shape_inference.infer_shapes(model)
    
    #print(op_name)
    #import code
    #code.interact(local=dict(globals(), **locals()))

    output_value_info = [value_info for value_info in model_out.graph.value_info if value_info.name == output_t_name] 
    if len(output_value_info) == 0:
        raise Exception(f'OP evaluation {op_name} failed with args: {input_tensors} and attrs: {attrs}')
    
    # create and return Tensor
    shape, dtype = vip_to_shape_dtype(output_value_info[0])
    output_t = Tensor(shape=shape, dtype=dtype, name=output_t_name)
    output_t._op_node_input_tensors = input_tensors
    output_t._op_node = node

    return output_t

def make_loop_body( input_iter_count : Tensor, input_is_running : Tensor, input_tensors : List[Tensor],
                    output_is_running : Tensor, output_tensors : List[Tensor], name=None ):
    """
    make a loop_body for loop operator
    """
    assert_tensor_shape_dtype(input_iter_count, [], np.int64)
    assert_tensor_shape_dtype(input_is_running, [], np.bool_)
    assert_tensor_shape_dtype(output_is_running, [], np.bool_)

    if len(input_tensors) != len(output_tensors):
        raise ValueError(f'len(input_tensors) {len(input_tensors)} != len(output_args){len(output_tensors)}')

    output_tensors = [output_is_running]+output_tensors
    tensor_list = [input_iter_count, input_is_running]+input_tensors+output_tensors
    nodes, input_vips = backward_gather(tensor_list)
    output_vips = [Tensor_to_vip(x) for x in output_tensors]

    if name is None:
        name = gen_unique_seq_name()
    graph_def = onnx_helper.make_graph(nodes, name, input_vips, output_vips)
    return graph_def

def make_model(outputs : List[Tensor]):
    nodes, input_vips = backward_gather(outputs)

    output_vips = [ Tensor_to_vip(x) for x in outputs]

    graph_def = onnx_helper.make_graph(nodes, 'graph', input_vips, output_vips)
    model = onnx_helper.make_model(graph_def)
    model.ir_version=7
    model.opset_import[0].version=get_opset_version()
    onnx.checker.check_model(model)

    return model
