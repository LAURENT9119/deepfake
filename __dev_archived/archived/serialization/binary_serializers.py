import io
import pickle
import struct
from pathlib import Path

import numpy as np

class BinarySerializer:
    """
    base class for binary serializers
    """

    def __init__(self):
        self.seek_stack = []

    # def to_bytes(self):
    #     """returns bytes() object containing copy of data from begin to current cursor"""
    #     size = self.tell()
    #     self.seek(0)
    #     return self.read_raw(size)
    def seek(self, offset, whence=0):
        """
        seek to offset

         whence     0(default) : from begin   , offset >= 0
                    1          : from current , offset any
                    2          : from end     , offset any

        returns cursor after seek
        """
        raise NotImplementedError()

    def tell(self):
        """returns current cursor offset"""
        raise NotImplementedError()
    
    def truncate(self):
        """truncate the serializer to the current cursor"""
        raise NotImplementedError()
    
    def fill_raw(self, byte, size : int):
        """fills a byte of size"""
        c_size = 16384
        n_count = size // c_size
        m_count = size % c_size
        f = bytes([byte]) * c_size
        
        if n_count >= 1:
            for _ in range(n_count):
                self.write_raw(f, c_size)        
        if m_count > 0:
            self.write_raw(f, m_count)
        
    def _write_raw_size(self, bytes_like, size=0, offset=0):
        if size == 0:
            if isinstance(bytes_like, memoryview):
                size = bytes_like.nbytes
            else:
                size = len(bytes_like)
            if offset != 0:
                size -= offset
        return size
        
    def write_raw (self, bytes_like, size=0, offset=0):
        """write bytes from bytes_like"""
        raise NotImplementedError()

    def read_raw(self, size):
        """read size amount of bytes and returns bytes() object"""
        raise NotImplementedError()

    def read_raw_into (self, bytes_like_or_io, size, offset=0):
        """read size amount of bytes into mutable bytes_like or io"""
        raise NotImplementedError()

    def get_raw(self, size, offset=None):
        """read size amount of bytes without increment the cursor and returns bytes() object"""
        c = self.tell()
        result = self.read_raw(size)
        self.seek(c)
        return result

    def push_seek(self, new_cursor=None, whence=0):
        """push cursor to stack and optionally set new"""
        self.seek_stack.append(self.tell())
        if new_cursor is not None:
            self.seek(new_cursor, whence)
            
    def pop_seek(self):
        """pop cursor from stack"""
        self.seek(self.seek_stack.pop())

    def write_bytes(self, b_bytes):
        """writes bytes() object"""
        self.write_fmt('Q', len(b_bytes))
        self.write_raw(b_bytes)

    def read_bytes(self):
        """reads bytes() object"""
        return self.read_raw( self.read_fmt('Q')[0] )

    def write_fmt_calc(self, fmt):
        """calc size for format"""
        return struct.calcsize(fmt)
        
    def write_fmt(self, fmt, *args):
        """write formatted data"""
        self.write_raw( struct.pack(fmt, *args) )

    def get_fmt(self, fmt):
        """read formatted data without incrementing a cursor"""
        return struct.unpack (fmt, self.get_raw(struct.calcsize(fmt)) )

    def read_fmt(self, fmt):
        """read formatted data"""
        return struct.unpack (fmt, self.read_raw(struct.calcsize(fmt)))

    def write_object(self, obj):
        """write object as pickled bytes"""
        self.write_bytes(pickle.dumps(obj))

    def read_object(self):
        """read pickled object"""
        return pickle.loads(self.read_bytes())

    def write_utf8(self, s):
        """write string as utf8"""
        self.write_bytes( s.encode('utf-8') )

    def read_utf8(self):
        """read string from utf8"""
        return self.read_bytes().decode('utf-8')

    def write_ndarray(self, npar : np.ndarray):
        """write np.ndarray"""
        np_view = memoryview(npar.reshape(-1))
        nbytes = np_view.nbytes

        self.write_object( (npar.shape, npar.dtype, nbytes, np_view.format) )
        self.write_raw(np_view, nbytes)

    def read_ndarray(self):
        """read np.ndarray"""
        shape, dtype, nbytes, format = self.read_object()
        np_ar = np.empty(shape, dtype)

        self.read_raw_into(memoryview(np_ar.reshape(-1)), nbytes)

        return np_ar

    def write_path(self, path : Path):
        """write Path object"""
        self.write_utf8(str(path))

    def read_path(self):
        """read Path object"""
        return Path(self.read_utf8())

class BinaryFileSerializer(BinarySerializer):

    def __init__(self, f):
        super().__init__()
        
        if not isinstance(f, io.IOBase):
            raise ValueError('f must be a file object.')
            
        if not f.readable() or not f.seekable():
            raise ValueError(f'file object {f} must be readable,seakable')

        self._f = f

    def seek(self, offset, whence=0):
        """
        seek to offset

         whence     0(default) : from begin   , offset >= 0
                    1          : from current , offset any
                    2          : from end     , offset any

        returns cursor after seek
        """
        f = self._f
        offset_max = f.seek(0,2)
        
        if whence == 1:
            offset += f.tell()
        elif whence == 2:
            offset += offset_max
            
        expand = offset - offset_max
        if expand > 0:
            # new offset will be more than file size
            # expand with zero bytes
            self.fill_raw(0x00, expand)

        return f.seek(offset,0)

    def tell(self):
        """returns current cursor offset"""
        return self._f.tell()
    
    def truncate(self):
        """truncate the file to the current cursor"""
        self._f.truncate()
    
    def write_raw(self, bytes_like, size=0, offset=0):
        """write bytes from bytes_like"""
        size = self._write_raw_size(bytes_like, size, offset)
        
        if isinstance(bytes_like, memoryview):
            bytes_like = bytes_like.cast('B')
        
        self._f.write( bytes_like[offset:offset+size] )

    def read_raw(self, size):
        """read size amount of bytes and returns bytes() object"""
        return self._f.read(size)

    def read_raw_into (self, bytes_like_or_io, size, offset=0):
        """read size amount of bytes into mutable bytes_like or io"""
        if isinstance(bytes_like_or_io, io.IOBase):
            b = self._f.read(size)
            bytes_like_or_io.write(b)
        else:
            self._f.read_into( memoryview(bytes_like_or_io).cast('B')[offset:offset+size] )

class BinaryBytesIOSerializer(BinaryFileSerializer):
    """"""
    def __init__(self, f=None):
        if f is None:
            f = io.BytesIO()
        super().__init__(f)


class BinaryMemoryViewSerializer(BinarySerializer):
    def __init__(self, mv : memoryview, offset=0):
        super().__init__()
        if offset != 0:
            mv = mv[offset:]
        self._mv = mv
        self._mv_size = mv.nbytes
        self._c = 0
        self._c_max = 0

    def seek(self, cursor, whence=0):
        """
        seek to offset

         whence     0(default) : from begin   , offset >= 0
                    1          : from current , offset any
                    2          : from end     , offset any

        returns cursor after seek
        """
        # memoryview is not expandable, thus just clip the cursor
        if whence == 0:
            self._c = min( max(cursor,0), self._mv_size)
            self._c_max = max(self._c, self._c_max)
        elif whence == 1:
            self._c = min( max(self._c + cursor, 0), self._mv_size)
            self._c_max = max(self._c, self._c_max)
        elif whence == 2:
            self._c = min( max(self._c_max + cursor, 0), self._mv_size)
            self._c_max = max(self._c, self._c_max)
        else:
            raise ValueError('whence != 0,1,2')
        
        return self._c

    def tell(self):
        """returns current cursor offset"""
        return self._c
    
    def truncate(self):
        """truncate the serializer to the current cursor"""
        self._c_max = self._c
        
    def write_raw (self, bytes_like, size=0, offset=0):
        """write bytes from bytes_like"""
        size = self._write_raw_size(bytes_like, size, offset)
        
        if isinstance(bytes_like, memoryview):
            bytes_like = bytes_like.cast('B')
        
        self._mv[self._c:self._c+size] = bytes_like[offset:offset+size]
        self._c += size
        
    def read_raw(self, size):
        result = self._mv[self._c:self._c+size].tobytes()
        self._c += size
        return result

    def read_raw_into (self, bytes_like_or_io, size, offset=0):
        """read size amount of bytes into mutable bytes_like or io"""
        if isinstance(bytes_like_or_io, io.IOBase):
            bytes_like_or_io.write( self._mv[self._c:self._c+size] )
        else:
            memoryview(bytes_like).cast('B')[offset:offset+size] = self._mv[self._c:self._c+size]
        self._c += size

    def write_fmt(self, fmt, *args):
        """write formatted data"""
        struct.pack_into(fmt, self._mv, self._c, *args )
        self._c += struct.calcsize(fmt)

    def get_fmt(self, fmt):
        """read formatted data without incrementing a cursor"""
        return struct.unpack_from(fmt, self._mv, self._c)

    def read_fmt(self, fmt):
        """read formatted data"""
        result = struct.unpack_from(fmt, self._mv, self._c)
        self._c += struct.calcsize(fmt)
        return result
