import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.autograd
import torch.functional as F
from localization import L, Localization
from modelhub import torch as torch_models
from modelhub.DFLive.DFMModel import get_available_devices
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from resources.fonts import QXFontDB
from xlib import torch as lib_torch
from xlib.console import diacon as dc
from xlib.torch.optim import AdaBelief
from xlib.torch.device import TorchDeviceInfo
from xlib import time as lib_time

from .TrainingDataGenerator import Data, TrainingDataGenerator


class CTSOTTrainerApp:
    def __init__(self, workspace_path : Path, faceset_path : Path):

        print('Initializing trainer.\n')
        print(f'Workspace path: {workspace_path}')
        print(f'Faceset path: {faceset_path}\n')

        workspace_path.mkdir(parents=True, exist_ok=True)
        self._workspace_path = workspace_path
        self._faceset_path = faceset_path

        self._is_viewing_data = False
        self._new_viewing_data = None
        self._new_preview_data : 'PreviewData' = None
        self._quit = False
        self._ctsot_lock = threading.Lock()
        
        self._device = None
        self._new_device = None
        
        print('Initializing...')
        
        self._training_generator = TrainingDataGenerator(faceset_path)
        
        self._model_data_path = model_data_path = workspace_path / 'model.dat'
        self._model_data = model_data = torch.load(model_data_path) if model_data_path.exists() else {}
        
        self.set_batch_size( model_data.get('batch_size', 4) )
        self.set_resolution( model_data.get('resolution', 224) )
        self.set_face_coverage_range( model_data.get('face_coverage_range', [2.3, 2.3]) )
        self._training_generator.set_running(True)
        
        device_info = model_data.get('device_index', None)
        if device_info is not None:
            device_info = lib_torch.get_device_info_by_index(device_info)
        if device_info is None:
            device_info = lib_torch.get_cpu_device_info()

        self.set_device_info(device_info)
        
        self.recreate_model(load=True)
        
        self._loss_history = model_data.get('loss_history', dict() )
        self.set_iteration( model_data.get('iteration', 0) )
        self.set_training( model_data.get('training', False) )

        # Console related
        self._ev_dlg_recreate_model = threading.Event()
        self._ev_dlg_preview_request = threading.Event()
        self._ev_dlg_save = threading.Event()
        self._ev_dlg_quit = threading.Event()

        self.get_main_dlg().set_current()

        

        threading.Thread(target=self.preview_thread_proc, daemon=True).start()

        self.main_loop()

        self._training_generator.kill()
    
    def recreate_model(self, load : bool = True):
        while True:
            ctsot = torch_models.CTSOTNet(resolution=self.get_resolution() )
            ctsot.train()
            if self._device is not None:
                ctsot.to(self._device)
            
            
            ctsot_optimizer = AdaBelief(ctsot.parameters(), lr=5e-5, lr_dropout=0.3)
            
            if load:
                ctsot_state_dict = self._model_data.get('ctsot_state_dict', None)
                if ctsot_state_dict is not None:
                    try:
                        ctsot.load_state_dict(ctsot_state_dict)
                        
                        ctsot_optimizer_state_dict = self._model_data.get('ctsot_optimizer_state_dict', None)
                        if ctsot_optimizer_state_dict is not None:
                            ctsot_optimizer.load_state_dict(ctsot_optimizer_state_dict)
                                
                    except:
                        print('Network weights have been reseted.')
                        self._model_data['ctsot_state_dict'] = None
                        self._model_data['ctsot_optimizer_state_dict'] = None
                        continue
            else:
                print('Network weights have been reseted.')
            break
                
        self._ctsot = ctsot
        self._ctsot_optimizer = ctsot_optimizer
        
    def set_device_info(self, device_info : TorchDeviceInfo):
        self._device_info, self._new_device = device_info, lib_torch.get_device(device_info)

    def get_batch_size(self) -> int: return self._batch_size
    def set_batch_size(self, batch_size : int):
        self._batch_size = batch_size
        self._training_generator.set_batch_size(batch_size)

    def get_resolution(self) -> int: return self._resolution
    def set_resolution(self, resolution : int):
        self._resolution = resolution
        # TODO dynamic change resolution with model. Training generator data with tags

        self._training_generator.set_resolution(resolution)

    def get_face_coverage_range(self) -> Tuple[float,float]: return self._face_coverage_range
    def set_face_coverage_range(self, face_coverage_range : Tuple[float,float] ):
        self._face_coverage_range = face_coverage_range
        self._training_generator.set_face_coverage_range(face_coverage_range)

    def get_iteration(self) -> int: return self._iteration
    def set_iteration(self, iteration : int):
        self._iteration = iteration
        
        rec_loss_history = self._loss_history.get('reconstruct', None)
        if rec_loss_history is not None:
            self._loss_history['reconstruct'] = rec_loss_history[:iteration]
                

    def get_training(self) -> bool: return self._training
    def set_training(self, training : bool):
        self._training = training

    def save(self):
        d = {'batch_size' : self.get_batch_size(),
             'resolution' : self.get_resolution(),
             'face_coverage_range' : self.get_face_coverage_range(),
             'iteration' : self.get_iteration(),
             'device_index' : None if self._device_info.is_cpu() else self._device_info.get_index(),
             'training' : self.get_training(),
             'loss_history' : self._loss_history,
             'ctsot_state_dict' : self._ctsot.state_dict(),
             'ctsot_optimizer_state_dict': self._ctsot_optimizer.state_dict(),
             }

        torch.save(d, self._model_data_path)

    def preview_thread_proc(self):
        
        
        while not self._quit:
            preview_data, self._new_preview_data = self._new_preview_data, None
            if preview_data is not None:
                # new preview data to show
                data = preview_data.training_data
                n = np.random.randint(data.batch_size)
                img1 = data.img1[n].transpose((1,2,0))
                img2 = data.img2[n].transpose((1,2,0))
                
                img1_ct_diff = data.img1_ct_diff[n].transpose((1,2,0))
                img1_ct = img1 + img1_ct_diff
                
                
                img1_ct_diff_pred = preview_data.img1_ct_diff_pred[n].transpose((1,2,0))
                img1_ct_pred = img1 + img1_ct_diff_pred
                
                screen = np.concatenate([img1, img2, img1_ct, (img1_ct_diff +1.0)/2.0, (img1_ct_diff_pred +1.0)/2.0, img1_ct_pred], 1)
                cv2.imshow('Preview', screen)
                
            
            viewing_data, self._new_viewing_data = self._new_viewing_data, None
            if viewing_data is not None:
                n = np.random.randint(viewing_data.batch_size)
                img1 = viewing_data.img1[n].transpose((1,2,0))
                img2 = viewing_data.img2[n].transpose((1,2,0))
                img1_ct_diff = viewing_data.img1_ct_diff[n].transpose((1,2,0))
                screen = np.concatenate([img1, img2, img1+img1_ct_diff, (img1_ct_diff+1.0)/2.0], 1)
                cv2.imshow('Viewing samples', screen)
                

            cv2.waitKey(5)
            #time.sleep(0.005)

    def main_loop(self):

        #self._training_generator.set_running(True)

        while not self._quit:
                
            if self._ev_dlg_recreate_model.is_set():
                self._ev_dlg_recreate_model.clear()
                self.recreate_model(load=False)
                
            if self._new_device is not None:
                # Handling new device request
                self._device, self._new_device = self._new_device, None
                self._ctsot.to(self._device)
                self._ctsot_optimizer.load_state_dict(self._ctsot_optimizer.state_dict())
                
            if self._training or self._is_viewing_data or self._ev_dlg_preview_request.is_set(): 
              training_data = self._training_generator.get_next_data(wait=False)
              if  training_data is not None and \
                  training_data.resolution == self.get_resolution(): # Skip if resolution is different, due to delay
                    
                    if self._training:
                        self._ctsot_optimizer.zero_grad()
                    
                    if self._ev_dlg_preview_request.is_set() or \
                       self._training:
                        # Inference for both preview and training
                        img1_t = torch.tensor(training_data.img1).to(self._device)
                        img2_t = torch.tensor(training_data.img2).to(self._device)
                        img1_ct_diff_t = torch.tensor(training_data.img1_ct_diff).to(self._device)
                        img1_ct_diff_pred_t = self._ctsot(img1_t, img2_t)
                    
                    if self._training:
                        # Training optimization step
                        loss_t = (img1_ct_diff_t-img1_ct_diff_pred_t).square().mean()*10.0
                        loss_t.backward()
                        self._ctsot_optimizer.step()
                            
                        loss = loss_t.detach().cpu().numpy()
                        
                        rec_loss_history = self._loss_history.get('reconstruct', None)
                        if rec_loss_history is None:
                            rec_loss_history = self._loss_history['reconstruct'] = []
                        rec_loss_history.append(loss)
                        
                        self.set_iteration( self.get_iteration() + 1 )
                    
                    if self._ev_dlg_preview_request.is_set():
                        self._ev_dlg_preview_request.clear()
                        # Preview request
                        pd = PreviewData()
                        pd.training_data = training_data
                        pd.img1_ct_diff_pred = img1_ct_diff_pred_t.detach().cpu().numpy()
                        self._new_preview_data = pd
                    
                    if self._is_viewing_data:
                        self._new_viewing_data = training_data

            if self._ev_dlg_save.is_set():
                self._ev_dlg_save.clear()
                print('Saving...')
                self.save()
                print('Saving done.')

                dc.Diacon.update_dlg()

            if self._ev_dlg_quit.is_set():
                self._ev_dlg_quit.clear()
                self._quit = True

            time.sleep(0.005)


    def get_main_dlg(self):
        last_loss = 'No data'
        rec_loss_history = self._loss_history.get('reconstruct', None)
        if rec_loss_history is not None:
            if len(rec_loss_history) != 0:
                last_loss = f'{rec_loss_history[-1]}'
        
        return dc.DlgChoices([
                # Resolution
                

                #dc.DlgChoice(name='training', row_def='| Training menu.',
                #            on_choose=lambda dlg: self.get_training_dlg(dlg).set_current()),
                            
                dc.DlgChoice(name='iteration', row_def=f'| Iteration | {self.get_iteration()}',
                         on_choose=lambda dlg: self.get_iteration_dlg(parent_dlg=dlg).set_current() ),
                
                dc.DlgChoice(name='l', row_def=f'| Loss history | {last_loss}',
                             on_choose=lambda dlg: dlg.recreate().set_current() ),
                
                dc.DlgChoice(name='device', row_def=f'| Set device | {self._device_info}',
                            on_choose=lambda dlg: self.get_training_device_dlg(dlg).set_current()),
                            
                dc.DlgChoice(name='samplegen', row_def='| Sample generator menu.',
                            on_choose=lambda dlg: self.get_sample_generator_dlg(dlg).set_current()),
                            
                dc.DlgChoice(name='p', row_def='| Show current preview.',
                            on_choose=lambda dlg: (self._ev_dlg_preview_request.set(), dlg.recreate().set_current())),
                
                dc.DlgChoice(name='training', row_def=f'| Training | {self._training}',
                         on_choose=lambda dlg: (self.set_training(not self.get_training()), dlg.recreate().set_current()) ),

                dc.DlgChoice(name='r', row_def='| Recreate model.',
                            on_choose=lambda dlg: (self._ev_dlg_recreate_model.set(), dlg.recreate().set_current()) ),

                dc.DlgChoice(name='save', row_def='| Save all.',
                            on_choose=lambda dlg: self._ev_dlg_save.set() ),

                dc.DlgChoice(name='quit', row_def='| Quit now.',
                            on_choose=lambda dlg: self._ev_dlg_quit.set() )
                ], on_recreate=lambda dlg: self.get_main_dlg(),
                top_rows_def='|c9 Main menu' )
        

    def get_sample_generator_dlg(self, parent_dlg):
        return dc.DlgChoices([

            dc.DlgChoice(name='fc', row_def=f'| Set face coverage',
                         on_choose=lambda dlg: dlg.recreate().set_current(),
                         ),

            dc.DlgChoice(name='vs', row_def=f'| Previewing samples | {self._is_viewing_data}',
                         on_choose=self.on_sample_generator_dlg_view_last_sample,
                         ),

            dc.DlgChoice(name='r', row_def=f'| Running | {self._training_generator.is_running()}',
                         on_choose=lambda dlg: (self._training_generator.set_running(not self._training_generator.is_running()), dlg.recreate().set_current()) ),

            ],
            on_recreate=lambda dlg: self.get_sample_generator_dlg(parent_dlg),
            on_back    =lambda dlg: parent_dlg.recreate().set_current(),
            top_rows_def='|c9 Sample generator menu' )

    def on_sample_generator_dlg_view_last_sample(self, dlg):
        self._is_viewing_data = not self._is_viewing_data
        dlg.recreate().set_current()

    def get_training_dlg(self, parent_dlg):
        return dc.DlgChoices([
            
            ],
            on_recreate=lambda dlg: self.get_training_dlg(parent_dlg),
            on_back    =lambda dlg: parent_dlg.recreate().set_current(),
            top_rows_def='|c9 Training menu' )

    def get_iteration_dlg(self, parent_dlg):
        return dc.DlgNumber(is_float=False, min_value=0,
                            on_value  = lambda dlg, value: (self.set_iteration(value), parent_dlg.recreate().set_current()),
                            on_recreate = lambda dlg: self.get_iteration_dlg(parent_dlg),
                            on_back   = lambda dlg: parent_dlg.recreate().set_current(),
                            top_rows_def='|c9 Set iteration',  )

    def get_training_device_dlg(self, parent_dlg):
        return DlgTorchDevicesInfo(on_device_choice = lambda dlg, device_info: (self.set_device_info(device_info), parent_dlg.recreate().set_current()),
                                   on_recreate = lambda dlg: self.get_training_device_dlg(parent_dlg),
                                   on_back     = lambda dlg: parent_dlg.recreate().set_current(),
                                   top_rows_def='|c9 Choose device'
                                   )


class DlgTorchDevicesInfo(dc.DlgChoices):
    def __init__(self, on_device_choice : Callable = None,
                       on_device_multi_choice : Callable = None,
                       on_recreate = None,
                       on_back : Callable = None,
                       top_rows_def : Union[str, List[str]] = None, 
                       bottom_rows_def : Union[str, List[str]] = None,):
        devices = lib_torch.get_available_devices_info()
        super().__init__(choices=[
                            dc.DlgChoice(name=f'{device.get_index()}' if not device.is_cpu() else 'cpu',
                                         row_def= f"| {str(device.get_name())} " +
                                                 (f"| {(device.get_total_memory() / 1024**3) :.3}Gb" if not device.is_cpu() else ""),
                                         on_choose= ( lambda dlg, i=i: on_device_choice(dlg, devices[i]) ) \
                                                      if on_device_choice is not None else None)
                            for i, device in enumerate(devices) ],
                         on_multi_choice=(lambda idxs: on_device_multi_choice([ devices[idx] for idx in idxs ])) \
                                          if on_device_multi_choice is not None else None,
                         on_recreate=on_recreate, on_back=on_back,
                         top_rows_def=top_rows_def, bottom_rows_def=bottom_rows_def)




class PreviewData:
    training_data : Data = None
    img1_ct_diff_pred : np.ndarray = None




    # import code
    # code.interact(local=dict(globals(), **locals()))
    # gen = TrainingDataGenerator(faceset_path)
    # gen.set_batch_size(2)
    # gen.set_resolution(256)
    # gen.set_face_coverage_range([2.3,2.3])
    # gen.set_running(True)
    # while True:
    #     x = gen.get_next_data()
    #     #import code
    #     #code.interact(local=dict(globals(), **locals()))

    #     #print('data ', x)
    #     #break
    #     cv2.imshow('', x[0][0])
    #     cv2.waitKey(0)
    # gen.kill()
    # time.sleep(0.5)
    # del gen
    # import gc
    # gc.collect()
    # gc.collect()
    # import code
    # code.interact(local=dict(globals(), **locals()))


