import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple, Union, Any

import cv2
import numpy as np
from xlib import face as lib_face
from xlib import mp as lib_mp
from xlib import mt as lib_mt
from xlib.image import color_transfer as lib_ct


class Data:
    def __init__(self):
        self.batch_size : int = None
        self.resolution : int = None
        self.img1 : np.ndarray = None
        self.img2 : np.ndarray = None
        self.img1_ct_diff : np.ndarray = None

class TrainingDataGenerator(lib_mp.SPMTWorker):
    def __init__(self, faceset_path : Path):
        faceset_path = Path(faceset_path)
        if not faceset_path.exists():
            raise Exception (f'{faceset_path} does not exist.')

        super().__init__(faceset_path)
        self._data = deque()
        self._running = False

    def get_next_data(self, wait : bool) -> Union[Data, None]:
        """
        wait and returns new generated data
        """

        while len(self._data) == 0:
            if not wait:
                return None
            time.sleep(0.005)

        x = self._data.popleft()
        self._send_msg('data_received')
        return x

    def is_running(self) -> bool: return self._running
    def set_running(self, running : bool):
        self._running = running
        self._send_msg('running', running)

    def set_batch_size(self, batch_size):
        self._send_msg('batch_size', batch_size)

    def set_resolution(self, resolution):
        self._send_msg('resolution', resolution)

    def set_face_coverage_range(self, face_coverage_range : Tuple[float, float]):
        self._send_msg('face_coverage_range', face_coverage_range)

    ###### IMPL HOST
    def _on_host_sub_message(self, name, *args, **kwargs):

        if name == 'data':
            self._data.append(args[0])
        elif name == 'running':
            self._running = args[0]

    ###### IMPL SUB
    def _on_sub_initialize(self, faceset_path : Path):
        self._faceset_path = faceset_path
        fs = lib_face.Faceset(faceset_path)

        # Gather all UFaceMark's uuids
        self._ufm_uuids = fs.get_all_UFaceMark_uuids()
        self._ufm_uuid_indexes = []
        self._fs = [None]*self._sub_get_thread_count()

        self._ufm_lock = threading.Lock()
        self._sent_buffers_atom = lib_mt.AtomicInteger()

        self._running = False

        self._batch_size = None
        self._resolution = None
        self._face_coverage_range = None

    def _on_sub_host_message(self, name, *args, **kwargs):
        """
        a message from host
        """
        if name == 'data_received':
            self._sent_buffers_atom.dec()
        elif name == 'batch_size':
            self._batch_size, = args
        elif name == 'resolution':
            self._resolution, = args
        elif name == 'face_coverage_range':
            self._face_coverage_range, = args
        elif name == 'running':
            running, = args
            if self._running != running:
                if running:
                    if self._fs is None:
                        print('Unable to start TrainingGenerator: faceset is not opened')
                        running = False
                    if self._batch_size is None:
                        print('Unable to start TrainingGenerator: batch_size must be set')
                        running = False
                    if self._resolution is None:
                        print('Unable to start TrainingGenerator: resolution must be set')
                        running = False
                    if self._face_coverage_range is None:
                        print('Unable to start TrainingGenerator: face_coverage_range must be set')
                        running = False

                self._running = running
                self._send_msg('running', running)

    ####### IMPL SUB THREAD
    # overridable
    def _on_sub_thread_initialize(self, thread_id):
        self._fs[thread_id] = lib_face.Faceset(self._faceset_path)
    def _on_sub_thread_finalize(self, thread_id):
        self._fs[thread_id].close()

    def _on_sub_thread_tick(self, thread_id):

        if self._running:
            if self._sent_buffers_atom.get_value() < self._sub_get_thread_count():
                fs = self._fs[thread_id]

                batch_size = self._batch_size
                resolution = self._resolution
                face_coverage = np.random.uniform(*self._face_coverage_range)

                rw_grid_cell_range = [3,7]
                rw_grid_rot_deg_range = [-180,180]
                rw_grid_scale_range = [-0.25, 0.25]
                rw_grid_tx_range = [-0.25, 0.25]
                rw_grid_ty_range = [-0.25, 0.25]

                align_rot_deg_range = [-15,15]
                align_scale_range = [-0.15, 0.15]
                align_tx_range = [-0.05, 0.05]
                align_ty_range = [-0.05, 0.05]

                img1_list = []
                img2_list = []
                img1_ct_diff_list = []

                for n in range(batch_size):

                    while True:
                        uuid1 = self._get_next_UFaceMark_uuid()
                        uuid2 = self._get_next_UFaceMark_uuid()

                        ufm1 = fs.get_UFaceMark_by_uuid(uuid1)
                        ufm2 = fs.get_UFaceMark_by_uuid(uuid2)

                        flmrks1 = ufm1.get_FLandmarks2D_best()
                        flmrks2 = ufm2.get_FLandmarks2D_best()
                        if flmrks1 is None:
                            print(f'Corrupted faceset, no FLandmarks2D for UFaceMark {ufm1.get_uuid()}')
                            continue
                        if flmrks2 is None:
                            print(f'Corrupted faceset, no FLandmarks2D for UFaceMark {ufm2.get_uuid()}')
                            continue

                        uimg1 = fs.get_UImage_by_uuid(ufm1.get_UImage_uuid())
                        uimg2 = fs.get_UImage_by_uuid(ufm2.get_UImage_uuid())
                        if uimg1 is None:
                            print(f'Corrupted faceset, no UImage for UFaceMark {ufm1.get_uuid()}')
                            continue
                        if uimg2 is None:
                            print(f'Corrupted faceset, no UImage for UFaceMark {ufm2.get_uuid()}')
                            continue

                        img1 = uimg1.get_image()
                        img2 = uimg2.get_image()

                        if img1 is None:
                            print(f'Corrupted faceset, no image in UImage {uimg1.get_uuid()}')
                            continue
                        if img2 is None:
                            print(f'Corrupted faceset, no image in UImage {uimg2.get_uuid()}')
                            continue

                        _, img_to_face_uni_mat1  = flmrks1.calc_cut( img1.shape[0:2], face_coverage, resolution)
                        _, img_to_face_uni_mat2  = flmrks2.calc_cut( img1.shape[0:2], face_coverage, resolution)


                        fw1 = lib_face.FaceWarper(img_to_face_uni_mat1,
                                                align_rot_deg=align_rot_deg_range,
                                                align_scale=align_scale_range,
                                                align_tx=align_tx_range,
                                                align_ty=align_ty_range,
                                                rw_grid_cell_count=rw_grid_cell_range,
                                                rw_grid_rot_deg=rw_grid_rot_deg_range,
                                                rw_grid_scale=rw_grid_scale_range,
                                                rw_grid_tx=rw_grid_tx_range,
                                                rw_grid_ty=rw_grid_ty_range,
                                                )
                        fw2 = lib_face.FaceWarper(img_to_face_uni_mat2,
                                                align_rot_deg=align_rot_deg_range,
                                                align_scale=align_scale_range,
                                                align_tx=align_tx_range,
                                                align_ty=align_ty_range,
                                                rw_grid_cell_count=rw_grid_cell_range,
                                                rw_grid_rot_deg=rw_grid_rot_deg_range,
                                                rw_grid_scale=rw_grid_scale_range,
                                                rw_grid_tx=rw_grid_tx_range,
                                                rw_grid_ty=rw_grid_ty_range,
                                                )

                        # gen random convex mask
                        mask = self._gen_random_convex_mask(3, 20, resolution)

                        new_img1 = fw1.transform(img1, resolution, random_warp=False).astype(np.float32) / 255.0
                        new_img2 = fw2.transform(img2, resolution, random_warp=False).astype(np.float32) / 255.0

                        new_img1_ct_diff = np.clip(lib_ct.sot(new_img1, new_img2, mask=mask, return_diff=True), -1, 1)

                        img1_list.append(new_img1*mask)
                        img2_list.append(new_img2*mask)
                        img1_ct_diff_list.append(new_img1_ct_diff*mask)

                        break

                data = Data()
                data.batch_size = batch_size
                data.resolution = resolution
                data.img1 = np.array(img1_list).transpose( (0,3,1,2))
                data.img2 = np.array(img2_list).transpose( (0,3,1,2))
                data.img1_ct_diff = np.array(img1_ct_diff_list).transpose( (0,3,1,2))



                self._send_msg('data', data)
                self._sent_buffers_atom.inc()

    def _gen_random_convex_mask(self, min_points, max_points, resolution):
        n_points = np.random.randint(min_points, max_points )
        pts = np.array( [ (np.random.randint(0, resolution), np.random.randint(0, resolution)) for n in range(n_points) ], np.int32)
        mask = np.zeros( (resolution,resolution,1), np.float32 )
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), (1,))
        return mask

    def _get_next_UFaceMark_uuid(self) -> bytes:
        with self._ufm_lock: #TODO
            if len(self._ufm_uuid_indexes) == 0:
                self._ufm_uuid_indexes = [*range(len(self._ufm_uuids))]
                np.random.shuffle(self._ufm_uuid_indexes)
            idx = self._ufm_uuid_indexes.pop()
            return self._ufm_uuids[idx]
