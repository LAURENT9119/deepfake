from ctypes import POINTER, c_uint, c_void_p, c_wchar_p
from enum import IntEnum
from typing import Union

from ..wintypes import (ERROR, GUID, HRESULT, REFGUID, REFIID, IUnknown,
                     dll_import, interface)
from .structs import *


@interface
class IDMLObject(IUnknown):
    def GetPrivateData          (self, guid : REFGUID, in_out_data_size : POINTER(c_uint), out_data : c_void_p ) -> HRESULT: ...
    def SetPrivateData          (self, guid : REFGUID, id : c_uint, data : c_void_p) -> HRESULT: ...
    def SetPrivateDataInterface (self, guid : REFGUID, interface : IUnknown) -> HRESULT: ...
    def SetName                 (self, name : c_wchar_p) -> HRESULT: ...
    IID = GUID('c8263aac-9e0c-4a2d-9b8e-007521a3317c')


@interface
class IDMLDevice(IDMLObject):
    
    
    # IFACEMETHOD(CheckFeatureSupport)(
    #     DML_FEATURE feature,
    #     UINT featureQueryDataSize,
    #     _In_reads_bytes_opt_(featureQueryDataSize) const void* featureQueryData,
    #     UINT featureSupportDataSize,
    #     _Out_writes_bytes_(featureSupportDataSize) void* featureSupportData
    #     ) = 0;
    
    # IFACEMETHOD(CreateOperator)(
    #     const DML_OPERATOR_DESC* desc,
    #     REFIID riid, // expected: IDMLOperator
    #     _COM_Outptr_opt_ void** ppv
    #     ) = 0;
    
    # IFACEMETHOD(CompileOperator)(
    #     IDMLOperator* op,
    #     DML_EXECUTION_FLAGS flags,
    #     REFIID riid, // expected: IDMLCompiledOperator
    #     _COM_Outptr_opt_ void** ppv
    #     ) = 0;
    
    # IFACEMETHOD(CreateOperatorInitializer)(
    #     UINT operatorCount,
    #     _In_reads_opt_(operatorCount) IDMLCompiledOperator* const* operators,
    #     REFIID riid, // expected: IDMLOperatorInitializer
    #     _COM_Outptr_ void** ppv
    #     ) = 0;
    
    # IFACEMETHOD(CreateCommandRecorder)(
    #     REFIID riid, // expected: IDMLCommandRecorder
    #     _COM_Outptr_ void** ppv
    #     ) = 0;
    
    # IFACEMETHOD(CreateBindingTable)(
    #     _In_opt_ const DML_BINDING_TABLE_DESC* desc,
    #     REFIID riid, // expected: IDMLBindingTable
    #     _COM_Outptr_ void** ppv
    #     ) = 0;
    
    # IFACEMETHOD(Evict)(
    #     UINT count,
    #     _In_reads_(count) IDMLPageable* const* ppObjects
    #     ) = 0;
    
    # IFACEMETHOD(MakeResident)(
    #     UINT count,
    #     _In_reads_(count) IDMLPageable* const* ppObjects
    #     ) = 0;
    
    # IFACEMETHOD(GetDeviceRemovedReason)(
    #     ) = 0;

    # IFACEMETHOD(GetParentDevice)(
    #     REFIID riid,
    #     _COM_Outptr_ void** ppv
    #     ) = 0;
    IID = GUID('6dbd6437-96fd-423f-a98c-ae5e7c2a573f')

@dll_import('directml')
def DMLCreateDevice(d3d12Device : IDMLDevice, flags : DML_CREATE_DEVICE_FLAGS, riid : REFIID, out_device : POINTER(IDMLDevice) ) -> HRESULT: ...
    
def DML_create_device(d3d12Device : IDMLDevice, flags : DML_CREATE_DEVICE_FLAGS) -> Union[IDMLDevice, None]:
    result = IDMLDevice()
    if DMLCreateDevice(d3d12Device, flags, IDMLDevice.IID, result) == ERROR.SUCCESS:
        return result
    return None
    