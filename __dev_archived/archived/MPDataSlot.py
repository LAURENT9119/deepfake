import io
import multiprocessing
import pickle
import time
from ..io import FormattedMemoryViewIO

from .MPAtomicInt32 import MPAtomicInt32
from .MPSharedMemory import MPSharedMemory

class MPDataSlot:
    """
    Multiprocess high performance multireader-multiwriter single data slot.
    MPDataSlot operates any picklable python object
    The slot mean only one object can exist in the slot in one time.
    """
    def __init__(self, size_mb):
        self._size = size = size_mb*1024*1024
        shared_mem = self._shared_mem = MPSharedMemory(size)
        self._atom = MPAtomicInt32(ar=shared_mem.get_ar(), index=0)
        
        self._avail_size = size-4-4-8
        self._last_pop_f = None
            
    def push(self, d):
        """
        push obj to the slot

        arguments

         d      picklable python object
         
        returns True if success, 
                otherwise False - the slot is not emptied by receiver side.
        """
        # Construct the data in local memory
        d_dumped = pickle.dumps(d, 4)
        size = len(d_dumped)
        
        if size >= self._avail_size:
            raise Exception('size of MPDataSlot is not enough to push the object')
        
        if self._atom.compare_exchange(0, 1) != 0:
            return False

        fmv = FormattedMemoryViewIO(self._shared_mem.get_mv()[4:])
        ver, = fmv.get_fmt('I')
        fmv.write_fmt('IQ', ver+1, size)
        fmv.write (d_dumped)

        self._atom.set(2, with_lock=False)

        return True

    def get_pop(self, your_ver):
        """
        get current or last data in the slot
        
        the data will not be popped.
        
        Also checks current ver with 'your_ver'
        
        returns
            obj, ver

        if nothing to get or ver the same, obj is None
        """
        fmv = FormattedMemoryViewIO(self._shared_mem.get_mv()[4:])
        ver, = fmv.read_fmt('I')
        if ver == 0 or ver == your_ver:
            return None, your_ver

        f = self._last_pop_f = io.BytesIO()
        
        while True:
            initial_val = self._atom.multi_compare_exchange( (0,2), 1)
            if initial_val in (0,2):
                break
            time.sleep(0.001)

        fmv.seek(0)
        ver, size = fmv.read_fmt('IQ')
        fmv.readinto(f, size )

        self._atom.set(initial_val, with_lock=False)
        
        f.seek(0)
        return pickle.load(f), ver


    def pop(self):
        """
        pop the MPDataSlotData
        returns

            MPDataSlotData or None
        """
        # Acquire the lock and copy the data to the local memory
        if self._atom.compare_exchange(2, 1) != 2:
            return None

        f = self._last_pop_f = io.BytesIO()
        fmv = FormattedMemoryViewIO(self._shared_mem.get_mv()[4:])
        ver, size = fmv.read_fmt('IQ')
        fmv.readinto(f, size )
        
        self._atom.set(0, with_lock=False)
        
        f.seek(0)
        
        return pickle.load(f)
    
    def get_last_pop(self):
        """
        get last popped data
        returns
            MPDataSlotData or None
        """
        f = self._last_pop_f
        if f is not None:
            f.seek(0)
            return pickle.load(f)
        return None

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_last_pop_f')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._last_pop_f = None
