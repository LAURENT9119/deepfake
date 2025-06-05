from ctypes import POINTER, c_uint, c_void_p, c_wchar_p
from enum import IntEnum
from typing import Union

from ..types import (ERROR, GUID, HRESULT, REFGUID, REFIID, IUnknown,
                     dll_import, interface)
from .structs import *


@interface
class ID3D12Object(IUnknown):
    def GetPrivateData          (self, guid : REFGUID, in_out_data_size : POINTER(c_uint), out_data : c_void_p ) -> HRESULT: ...
    def SetPrivateData          (self, guid : REFGUID, id : c_uint, data : c_void_p) -> HRESULT: ...
    def SetPrivateDataInterface (self, guid : REFGUID, interface : IUnknown) -> HRESULT: ...
    def SetName                 (self, name : c_wchar_p) -> HRESULT: ...
    IID = GUID('c4fec28f-7966-4e95-9f94-f431cb56c3b8')
      
@interface
class ID3D12DeviceChild(ID3D12Object):
    def GetDevice(self, riid : REFIID, out_device : POINTER(IUnknown)) -> HRESULT: ...
    IID = GUID('905db94b-a00c-4140-9df5-2b64ca9ea357')
        
@interface
class ID3D12Pageable(ID3D12DeviceChild):
    IID = GUID('63ee58fb-1268-4835-86da-f008ce62f0d6')
        
@interface
class ID3D12CommandQueue(ID3D12Pageable):
    # virtual void STDMETHODCALLTYPE UpdateTileMappings( 
    #     _In_  ID3D12Resource *pResource,
    #     UINT NumResourceRegions,
    #     _In_reads_opt_(NumResourceRegions)  const D3D12_TILED_RESOURCE_COORDINATE *pResourceRegionStartCoordinates,
    #     _In_reads_opt_(NumResourceRegions)  const D3D12_TILE_REGION_SIZE *pResourceRegionSizes,
    #     _In_opt_  ID3D12Heap *pHeap,
    #     UINT NumRanges,
    #     _In_reads_opt_(NumRanges)  const D3D12_TILE_RANGE_FLAGS *pRangeFlags,
    #     _In_reads_opt_(NumRanges)  const UINT *pHeapRangeStartOffsets,
    #     _In_reads_opt_(NumRanges)  const UINT *pRangeTileCounts,
    #     D3D12_TILE_MAPPING_FLAGS Flags) = 0;
    
    # virtual void STDMETHODCALLTYPE CopyTileMappings( 
    #     _In_  ID3D12Resource *pDstResource,
    #     _In_  const D3D12_TILED_RESOURCE_COORDINATE *pDstRegionStartCoordinate,
    #     _In_  ID3D12Resource *pSrcResource,
    #     _In_  const D3D12_TILED_RESOURCE_COORDINATE *pSrcRegionStartCoordinate,
    #     _In_  const D3D12_TILE_REGION_SIZE *pRegionSize,
    #     D3D12_TILE_MAPPING_FLAGS Flags) = 0;
    
    # virtual void STDMETHODCALLTYPE ExecuteCommandLists( 
    #     _In_  UINT NumCommandLists,
    #     _In_reads_(NumCommandLists)  ID3D12CommandList *const *ppCommandLists) = 0;
    
    # virtual void STDMETHODCALLTYPE SetMarker( 
    #     UINT Metadata,
    #     _In_reads_bytes_opt_(Size)  const void *pData,
    #     UINT Size) = 0;
    
    # virtual void STDMETHODCALLTYPE BeginEvent( 
    #     UINT Metadata,
    #     _In_reads_bytes_opt_(Size)  const void *pData,
    #     UINT Size) = 0;
    
    # virtual void STDMETHODCALLTYPE EndEvent( void) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE Signal( 
    #     ID3D12Fence *pFence,
    #     UINT64 Value) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE Wait( 
    #     ID3D12Fence *pFence,
    #     UINT64 Value) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE GetTimestampFrequency( 
    #     _Out_  UINT64 *pFrequency) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE GetClockCalibration( 
    #     _Out_  UINT64 *pGpuTimestamp,
    #     _Out_  UINT64 *pCpuTimestamp) = 0;
    
    # virtual D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc( void) = 0;
    
    IID = GUID('0ec870a6-5d7e-4c22-8cfc-5baae07616ed')
    
@interface
class ID3D12CommandAllocator(ID3D12Pageable):
    def Reset(self) -> HRESULT: ...
    IID = GUID('6102dee4-af59-4b09-b999-b44d73f09b24')

@interface
class ID3D12CommandList(ID3D12DeviceChild):
    def GetType(self) -> D3D12_COMMAND_LIST_TYPE: ...
    IID = GUID('7116d91c-e7e4-47ce-b8c6-ec8168f437e5')
    
@interface
class ID3D12Device(ID3D12Object):
    def GetNodeCount(self) -> c_uint: ...
    def CreateCommandQueue(self, desc : D3D12_COMMAND_QUEUE_DESC, riid : REFIID, out_command_queue : POINTER(ID3D12CommandQueue) ) -> HRESULT: ...
    def CreateCommandAllocator( self, type : D3D12_COMMAND_LIST_TYPE, riid : REFIID, out_command_allocator : POINTER(ID3D12CommandAllocator) ) -> HRESULT: ...
    def CreateGraphicsPipelineState(self) -> None: ...
    def CreateComputePipelineState(self) -> None: ...
    def CreateCommandList(self, nodeMask : c_uint, type : D3D12_COMMAND_LIST_TYPE, command_allocator : ID3D12CommandAllocator, pInitialState : c_void_p, riid : REFIID, out_command_list : POINTER(ID3D12CommandList) ) -> HRESULT: ...
    
    # virtual HRESULT STDMETHODCALLTYPE CheckFeatureSupport( 
    #     D3D12_FEATURE Feature,
    #     _Inout_updates_bytes_(FeatureSupportDataSize)  void *pFeatureSupportData,
    #     UINT FeatureSupportDataSize) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE CreateDescriptorHeap( 
    #     _In_  const D3D12_DESCRIPTOR_HEAP_DESC *pDescriptorHeapDesc,
    #     REFIID riid,
    #     _COM_Outptr_  void **ppvHeap) = 0;
    
    # virtual UINT STDMETHODCALLTYPE GetDescriptorHandleIncrementSize( 
    #     _In_  D3D12_DESCRIPTOR_HEAP_TYPE DescriptorHeapType) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE CreateRootSignature( 
    #     _In_  UINT nodeMask,
    #     _In_reads_(blobLengthInBytes)  const void *pBlobWithRootSignature,
    #     _In_  SIZE_T blobLengthInBytes,
    #     REFIID riid,
    #     _COM_Outptr_  void **ppvRootSignature) = 0;
    
    # virtual void STDMETHODCALLTYPE CreateConstantBufferView( 
    #     _In_opt_  const D3D12_CONSTANT_BUFFER_VIEW_DESC *pDesc,
    #     _In_  D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) = 0;
    
    # virtual void STDMETHODCALLTYPE CreateShaderResourceView( 
    #     _In_opt_  ID3D12Resource *pResource,
    #     _In_opt_  const D3D12_SHADER_RESOURCE_VIEW_DESC *pDesc,
    #     _In_  D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) = 0;
    
    # virtual void STDMETHODCALLTYPE CreateUnorderedAccessView( 
    #     _In_opt_  ID3D12Resource *pResource,
    #     _In_opt_  ID3D12Resource *pCounterResource,
    #     _In_opt_  const D3D12_UNORDERED_ACCESS_VIEW_DESC *pDesc,
    #     _In_  D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) = 0;
    
    # virtual void STDMETHODCALLTYPE CreateRenderTargetView( 
    #     _In_opt_  ID3D12Resource *pResource,
    #     _In_opt_  const D3D12_RENDER_TARGET_VIEW_DESC *pDesc,
    #     _In_  D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) = 0;
    
    # virtual void STDMETHODCALLTYPE CreateDepthStencilView( 
    #     _In_opt_  ID3D12Resource *pResource,
    #     _In_opt_  const D3D12_DEPTH_STENCIL_VIEW_DESC *pDesc,
    #     _In_  D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) = 0;
    
    # virtual void STDMETHODCALLTYPE CreateSampler( 
    #     _In_  const D3D12_SAMPLER_DESC *pDesc,
    #     _In_  D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) = 0;
    
    # virtual void STDMETHODCALLTYPE CopyDescriptors( 
    #     _In_  UINT NumDestDescriptorRanges,
    #     _In_reads_(NumDestDescriptorRanges)  const D3D12_CPU_DESCRIPTOR_HANDLE *pDestDescriptorRangeStarts,
    #     _In_reads_opt_(NumDestDescriptorRanges)  const UINT *pDestDescriptorRangeSizes,
    #     _In_  UINT NumSrcDescriptorRanges,
    #     _In_reads_(NumSrcDescriptorRanges)  const D3D12_CPU_DESCRIPTOR_HANDLE *pSrcDescriptorRangeStarts,
    #     _In_reads_opt_(NumSrcDescriptorRanges)  const UINT *pSrcDescriptorRangeSizes,
    #     _In_  D3D12_DESCRIPTOR_HEAP_TYPE DescriptorHeapsType) = 0;
    
    # virtual void STDMETHODCALLTYPE CopyDescriptorsSimple( 
    #     _In_  UINT NumDescriptors,
    #     _In_  D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptorRangeStart,
    #     _In_  D3D12_CPU_DESCRIPTOR_HANDLE SrcDescriptorRangeStart,
    #     _In_  D3D12_DESCRIPTOR_HEAP_TYPE DescriptorHeapsType) = 0;
    
    # virtual D3D12_RESOURCE_ALLOCATION_INFO STDMETHODCALLTYPE GetResourceAllocationInfo( 
    #     _In_  UINT visibleMask,
    #     _In_  UINT numResourceDescs,
    #     _In_reads_(numResourceDescs)  const D3D12_RESOURCE_DESC *pResourceDescs) = 0;
    
    # virtual D3D12_HEAP_PROPERTIES STDMETHODCALLTYPE GetCustomHeapProperties( 
    #     _In_  UINT nodeMask,
    #     D3D12_HEAP_TYPE heapType) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE CreateCommittedResource( 
    #     _In_  const D3D12_HEAP_PROPERTIES *pHeapProperties,
    #     D3D12_HEAP_FLAGS HeapFlags,
    #     _In_  const D3D12_RESOURCE_DESC *pDesc,
    #     D3D12_RESOURCE_STATES InitialResourceState,
    #     _In_opt_  const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    #     REFIID riidResource,
    #     _COM_Outptr_opt_  void **ppvResource) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE CreateHeap( 
    #     _In_  const D3D12_HEAP_DESC *pDesc,
    #     REFIID riid,
    #     _COM_Outptr_opt_  void **ppvHeap) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE CreatePlacedResource( 
    #     _In_  ID3D12Heap *pHeap,
    #     UINT64 HeapOffset,
    #     _In_  const D3D12_RESOURCE_DESC *pDesc,
    #     D3D12_RESOURCE_STATES InitialState,
    #     _In_opt_  const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    #     REFIID riid,
    #     _COM_Outptr_opt_  void **ppvResource) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE CreateReservedResource( 
    #     _In_  const D3D12_RESOURCE_DESC *pDesc,
    #     D3D12_RESOURCE_STATES InitialState,
    #     _In_opt_  const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    #     REFIID riid,
    #     _COM_Outptr_opt_  void **ppvResource) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE CreateSharedHandle( 
    #     _In_  ID3D12DeviceChild *pObject,
    #     _In_opt_  const SECURITY_ATTRIBUTES *pAttributes,
    #     DWORD Access,
    #     _In_opt_  LPCWSTR Name,
    #     _Out_  HANDLE *pHandle) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE OpenSharedHandle( 
    #     _In_  HANDLE NTHandle,
    #     REFIID riid,
    #     _COM_Outptr_opt_  void **ppvObj) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE OpenSharedHandleByName( 
    #     _In_  LPCWSTR Name,
    #     DWORD Access,
    #     /* [annotation][out] */ 
    #     _Out_  HANDLE *pNTHandle) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE MakeResident( 
    #     UINT NumObjects,
    #     _In_reads_(NumObjects)  ID3D12Pageable *const *ppObjects) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE Evict( 
    #     UINT NumObjects,
    #     _In_reads_(NumObjects)  ID3D12Pageable *const *ppObjects) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE CreateFence( 
    #     UINT64 InitialValue,
    #     D3D12_FENCE_FLAGS Flags,
    #     REFIID riid,
    #     _COM_Outptr_  void **ppFence) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE GetDeviceRemovedReason( void) = 0;
    
    # virtual void STDMETHODCALLTYPE GetCopyableFootprints( 
    #     _In_  const D3D12_RESOURCE_DESC *pResourceDesc,
    #     _In_range_(0,D3D12_REQ_SUBRESOURCES)  UINT FirstSubresource,
    #     _In_range_(0,D3D12_REQ_SUBRESOURCES-FirstSubresource)  UINT NumSubresources,
    #     UINT64 BaseOffset,
    #     _Out_writes_opt_(NumSubresources)  D3D12_PLACED_SUBRESOURCE_FOOTPRINT *pLayouts,
    #     _Out_writes_opt_(NumSubresources)  UINT *pNumRows,
    #     _Out_writes_opt_(NumSubresources)  UINT64 *pRowSizeInBytes,
    #     _Out_opt_  UINT64 *pTotalBytes) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE CreateQueryHeap( 
    #     _In_  const D3D12_QUERY_HEAP_DESC *pDesc,
    #     REFIID riid,
    #     _COM_Outptr_opt_  void **ppvHeap) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE SetStablePowerState( 
    #     BOOL Enable) = 0;
    
    # virtual HRESULT STDMETHODCALLTYPE CreateCommandSignature( 
    #     _In_  const D3D12_COMMAND_SIGNATURE_DESC *pDesc,
    #     _In_opt_  ID3D12RootSignature *pRootSignature,
    #     REFIID riid,
    #     _COM_Outptr_opt_  void **ppvCommandSignature) = 0;
    
    # virtual void STDMETHODCALLTYPE GetResourceTiling( 
    #     _In_  ID3D12Resource *pTiledResource,
    #     _Out_opt_  UINT *pNumTilesForEntireResource,
    #     _Out_opt_  D3D12_PACKED_MIP_INFO *pPackedMipDesc,
    #     _Out_opt_  D3D12_TILE_SHAPE *pStandardTileShapeForNonPackedMips,
    #     _Inout_opt_  UINT *pNumSubresourceTilings,
    #     _In_  UINT FirstSubresourceTilingToGet,
    #     _Out_writes_(*pNumSubresourceTilings)  D3D12_SUBRESOURCE_TILING *pSubresourceTilingsForNonPackedMips) = 0;
    
    # virtual LUID STDMETHODCALLTYPE GetAdapterLuid( void) = 0;
    
    IID = GUID('189819f1-1db6-4b57-be54-1821339b85f7')


    def create_command_queue(self, desc : D3D12_COMMAND_QUEUE_DESC) -> Union[ID3D12CommandQueue, None]:
        result = ID3D12CommandQueue()
        if self.CreateCommandQueue(desc, ID3D12CommandQueue.IID,  result) == ERROR.SUCCESS:
            return result
        return None
        
    def create_command_allocator(self, type : D3D12_COMMAND_LIST_TYPE) -> Union[ID3D12CommandAllocator, None]:
        result = ID3D12CommandAllocator()
        if self.CreateCommandAllocator(type, ID3D12CommandAllocator.IID,  result) == ERROR.SUCCESS:
            return result
        return None    

    def create_command_list(self, nodeMask : c_uint, type : D3D12_COMMAND_LIST_TYPE, command_allocator : ID3D12CommandAllocator) -> Union[ID3D12CommandList, None]:
        result = ID3D12CommandList()
        
        if self.CreateCommandList(nodeMask, type, command_allocator, 0, ID3D12CommandList.IID, result) == ERROR.SUCCESS:
            return result
        return None
    

@dll_import('d3d12')    
def D3D12CreateDevice(adapter : IUnknown, MinimumFeatureLevel : D3D_FEATURE_LEVEL, riid : REFIID, out_device : POINTER(ID3D12Device)) -> HRESULT: ...

def d3d12_create_device(adapter : IUnknown, MinimumFeatureLevel : D3D_FEATURE_LEVEL) -> Union[ID3D12Device, None]:
    result = ID3D12Device()
    if D3D12CreateDevice(adapter, MinimumFeatureLevel, ID3D12Device.IID, result) == ERROR.SUCCESS:
        return result
    return None
    