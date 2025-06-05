import time
from pathlib import Path

from xlib import os as lib_os
from xlib.facemeta import Faceset
from xlib.mp import csw as lib_csw

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendWeakHeap, BackendWorker, BackendWorkerState)

##### CURRENTLY UNUSED


class FacesetSaver(BackendHost):
    def __init__(self, weak_heap : BackendWeakHeap, bc_in : BackendConnection, bc_out : BackendConnection, backend_db : BackendDB = None):
        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=FacesetSaverWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, bc_in, bc_out], )

    def get_control_sheet(self) -> 'Sheet.Host': return super().get_control_sheet()

class FacesetSaverWorker(BackendWorker):
    def get_state(self) -> 'WorkerState': return super().get_state()
    def get_control_sheet(self) -> 'Sheet.Worker': return super().get_control_sheet()

    def on_start(self, weak_heap : BackendWeakHeap,  bc_in : BackendConnection,  bc_out : BackendConnection):
        self.weak_heap = weak_heap
        self.bc_in = bc_in
        self.bc_out = bc_out
        self.pending_bcd = None
        self.faceset = None

        lib_os.set_timer_resolution(1)

        state, cs = self.get_state(), self.get_control_sheet()
        cs.faceset_path.call_on_paths(self.on_cs_faceset_path)
        cs.reload_signal.call_on_signal(self.on_reload_signal)


        cs.faceset_path.enable()
        cs.faceset_path.set_config( lib_csw.Paths.Config.Directory(caption='Faceset directory') )
        cs.faceset_path.set_paths(state.faceset_path)


    def set_cs_error(self, err=None):
        if err is None:
            err = ""

        cs = self.get_control_sheet()
        if len(err) == 0:
            cs.error.disable()
        else:
            cs.error.enable()
        cs.error.set_text(err)

    def on_cs_faceset_path(self, paths):
        self.set_cs_error(None)
        state, cs = self.get_state(), self.get_control_sheet()

        faceset_path = paths[0] if len(paths) != 0 else None
        cs.reload_signal.disable()
        cs.faces_count.disable()

        if faceset_path is not None:
            err = None
            try:
                faceset = Faceset(faceset_path)
            except Exception as e:
                err = str(e)

            if err is not None:
                faceset_path = None
                cs.faceset_path.set_paths([])
                self.set_cs_error(err)
            else:
                self.faceset = faceset

                cs.reload_signal.enable()

                self.update_faces_count()
                face_count = faceset.get_face_count()
                cs.faces_count.enable()
                cs.faces_count.set_config(lib_csw.Number.Config(max=face_count, read_only=True))
                cs.faces_count.set_number(face_count)



            state.faceset_path = faceset_path

    def update_faces_count(self):
        state, cs = self.get_state(), self.get_control_sheet()
        face_count = self.faceset.get_face_count()
        cs.faces_count.enable()
        cs.faces_count.set_config(lib_csw.Number.Config(max=face_count, read_only=True))
        cs.faces_count.set_number(face_count)


    def on_reload_signal(self):
        state, cs = self.get_state(), self.get_control_sheet()

        faceset = self.faceset
        if faceset is not None:
            try:
                faceset.reload()
                self.update_faces_count()
            except Exception as e:
                cs.faceset_path.set_paths([])
                self.set_cs_error(str(e))

    def on_tick(self):
        state, cs = self.get_state(), self.get_control_sheet()

        if self.pending_bcd is None:
            self.start_profile_timing()

            bcd = self.bc_in.read(timeout=0.005)
            if bcd is not None:
                bcd.assign_weak_heap(self.weak_heap)

                faceset, faces = self.faceset, bcd.get_faces()
                if faceset is not None and faces is not None and len(faces) != 0:
                    for face in faces:
                        faceset.save_face(face)

                        aligned_name = face.get_aligned_name()
                        faceset.save_image(aligned_name, bcd.get_image(aligned_name) )

                    self.update_faces_count()

                self.stop_profile_timing()
                self.pending_bcd = bcd

        if self.pending_bcd is not None:
            if self.bc_out.is_full_read(1):
                self.bc_out.write(self.pending_bcd)
                self.pending_bcd = None
            else:
                time.sleep(0.001)

class Sheet:
    class Host(lib_csw.Sheet.Host):
        def __init__(self):
            super().__init__()
            self.error = lib_csw.Text.Client()
            self.faceset_path = lib_csw.Paths.Client()
            self.faces_count = lib_csw.Number.Client()
            self.reload_signal = lib_csw.Signal.Client()

    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.error = lib_csw.Text.Host()
            self.faceset_path = lib_csw.Paths.Host()
            self.faces_count = lib_csw.Number.Host()
            self.reload_signal = lib_csw.Signal.Host()

class WorkerState(BackendWorkerState):
    faceset_path : Path = None
