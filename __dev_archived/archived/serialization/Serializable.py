import numpy as np
import pickle
import zlib
from pathlib import Path

from .binary_serializers import BinarySerializer

class Serializable:
    """
    Serializable is a base class for automatically and fast
    serialization internal data to seekable containers, such as file or memoryview

    Serializable is a more convenient way to store/load the classes with the data.

    Serializable class must not contain any args in __init__(), but kwargs are possible

    advantages:
     + serialize directly to the shared memory or disk file

     + class can be expanded before deserialization (new internal variables)

    notes:
     var_names with _ at the start and at the end will not be serialized
     
     to achieve maximum performance,
     each nested Serializable must have only one reference from parents,
     otherwise they will be deserialized as different objects

    """
    

    def __init__(self, *args, **kwargs):
        ...
        
    def _on_serialize_state(self, hint_compact=False):
        """
        overridable
        return the dict of variables which should be serialized
        
        if hint_compact==True, 
        you can for example manually compress the image to the png format
        """
        return self.__dict__.copy()
        
    def _on_deserialize_state(self, state, hint_compact=False):
        """
        overridable
        apply state dict to the object
        
        the behaviour as pickling __setstate__
        the difference is the object is constructed() before.
        
        if hint_compact==True, 
        you should apply decompress methods if something was compressed before
        """
        self.__dict__.update(state)
        

    def __getstate__(self):
        d = self.__dict__.copy()
        return d
        
    def __setstate__(self, d):
        
        self.__dict__.update(d)
        
    # def __repr__(self): return self.__str__()
    # def __str__(self):
    #     s = ""

    @staticmethod
    def _serialize_nested(parent : 'Serializable', obj, name : str, bf : BinarySerializer, hint_compact : bool):

        if obj is None:
            bf.write_fmt('B', 0)
        elif isinstance(obj, (bytes, bytearray) ):
            bf.write_fmt('B', 2 if isinstance(obj, bytes) else 3)
            bf.write_bytes(obj)
        elif isinstance(obj, np.ndarray):
            bf.write_fmt('B', 4)
            bf.write_ndarray(obj)
        elif isinstance(obj, Path):
            bf.write_fmt('B', 5)
            bf.write_path(obj)
        elif isinstance(obj, Serializable):
            bf.write_fmt('B', 6)
            Serializable.serialize(obj, bf, hint_compact)
        elif isinstance(obj, (list,tuple)):
            bf.write_fmt('B', 7 if isinstance(obj, list) else 8)
            bf.write_fmt('Q', len(obj))
            for list_obj in obj:
                Serializable._serialize_nested(parent, list_obj, None, bf, hint_compact)
        elif isinstance(obj, dict):
            bf.write_fmt('B', 9)
            bf.write_fmt('Q', len(obj))
            for key in obj:
                bf.write_bytes(pickle.dumps(key))
                Serializable._serialize_nested(parent, obj[key], None, bf, hint_compact)
        else:
            bf.write_fmt('B', 1)
            bf.write_bytes(pickle.dumps(obj))

    #@staticmethod
    def serialize(obj, bf : BinarySerializer, hint_compact=False):
        """
        
         hint_compact(False)    bool    hint to serialize in compact mode where possible
        """
        if not isinstance(obj, Serializable):
            raise ValueError('obj must be an instance of Serializable')

        bf.write_fmt('III?', 0xFEEDCAFE, 0, 0, hint_compact) # magic number, serializer version, reserved
        
        bf.write_bytes(pickle.dumps(obj.__class__))
        vars_count_cursor = bf.tell()
        bf.write_fmt('I', 0)

        vars_count = 0
        
        obj_vars = vars(obj)
        
        # get_state_func = getattr(obj, '__getstate__', None)
        # if get_state_func is not None:
        #     obj_vars = get_state_func()
        # else:
        #     obj_vars = vars(obj)
        
        for v_name in obj_vars:
            if len(v_name) >= 2 and v_name[0] == '_' and v_name[-1] == '_':
                continue

            bf.write_utf8(v_name)
            Serializable._serialize_nested( obj, obj_vars[v_name], v_name, bf, hint_compact)
            vars_count += 1
        
        bf.push_seek(vars_count_cursor)
        bf.write_fmt('I', vars_count)
        bf.pop_seek()

    @staticmethod
    def _deserialize_nested(parent : 'Serializable', name : str, bf : BinarySerializer, hint_compact : bool):
        obj_type, = bf.read_fmt('B')        
        if obj_type == 0:
            obj = None
        elif obj_type == 1:
            obj = pickle.loads(bf.read_bytes())
        elif obj_type >= 2 and obj_type <= 3:
            obj = bf.read_bytes()
            if obj_type == 3:
                obj = bytearray(obj)
        elif obj_type == 4:
            obj = bf.read_ndarray()
        elif obj_type == 5:
            obj = bf.read_path()
        elif obj_type == 6:
            obj = Serializable.deserialize(bf)
        elif obj_type >= 7 and obj_type <= 8:
            list_len, = bf.read_fmt('Q')
            obj = []
            for i in range(list_len):
                obj.append( Serializable._deserialize_nested(parent, None, bf, hint_compact) )
            if obj_type == 8:
                obj = tuple(obj)
        elif obj_type == 9:
            list_len, = bf.read_fmt('Q')
            obj = {}
            for i in range(list_len):
                key = pickle.loads(bf.read_bytes())
                obj[key] = Serializable._deserialize_nested(parent, None, bf, hint_compact)
        else:
            raise Exception(f'unknown type of obj: {obj_type}')

                
        return obj

    @staticmethod
    def deserialize(bf : BinarySerializer):
        magic_num, _, _, hint_compact = bf.read_fmt('III?')
        
        if magic_num != 0xFEEDCAFE:
            raise Exception(magic_num,'Data is corrupted.')
        
        cls_ = pickle.loads(bf.read_bytes())
        obj = cls_()

        vars_count, = bf.read_fmt('I')
        for i in range(vars_count):
            var_name = bf.read_utf8()
            var_obj = Serializable._deserialize_nested(obj, var_name, bf, hint_compact)
            setattr(obj, var_name, var_obj)

        return obj