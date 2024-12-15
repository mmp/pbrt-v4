#ifndef PBRT_UTIL_HIP_ALIASES_H
#define PBRT_UTIL_HIP_ALIASES_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_gl_interop.h>

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString

#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceGetLimit hipDeviceGetLimit
#define cudaDeviceSetLimit hipDeviceSetLimit
#define cudaDeviceSetCacheConfig hipDeviceSetCacheConfig
#define cudaLimitStackSize hipLimitStackSize
#define cudaLimitPrintfFifoSize hipLimitPrintfFifoSize
#define cudaFuncCachePreferL1 hipFuncCachePreferL1
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDevAttrKernelExecTimeout hipDeviceAttributeKernelExecTimeout
#define cudaDevAttrConcurrentManagedAccess hipDeviceAttributeConcurrentManagedAccess
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion

#define cudaGraphicsMapResources hipGraphicsMapResources
#define cudaGraphicsUnmapResources hipGraphicsUnmapResources
#define cudaGraphicsResourceGetMappedPointer hipGraphicsResourceGetMappedPointer
#define cudaGraphicsResource hipGraphicsResource 
#define cudaGraphicsGLRegisterBuffer hipGraphicsGLRegisterBuffer
#define cudaGraphicsMapResources hipGraphicsMapResources
#define cudaGraphicsMapFlagsWriteDiscard hipGraphicsRegisterFlagsWriteDiscard

#define cudaGLGetDevices hipGLGetDevices
#define cudaGLDeviceListAll hipGLDeviceListAll

#define CUdeviceptr hipDeviceptr_t
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaFree hipFree

#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaMemset hipMemset
#define cudaMemAdvise hipMemAdvise
#define cudaMemAdviseSetReadMostly hipMemAdviseSetReadMostly
#define cudaMemAdviseSetPreferredLocation hipMemAdviseSetPreferredLocation

#define cudaArray_t hipArray_t
#define cudaMallocArray hipMallocArray
#define cudaMemcpy2DToArray hipMemcpy2DToArray
#define cudaMipmappedArray_t  hipMipmappedArray_t
#define cudaMallocMipmappedArray hipMallocMipmappedArray
#define cudaGetMipmappedArrayLevel hipGetMipmappedArrayLevel

#define cudaExtent hipExtent
#define make_cudaExtent make_hipExtent

#define cudaTextureObject_t hipTextureObject_t
#define cudaCreateTextureObject hipCreateTextureObject
#define cudaChannelFormatDesc hipChannelFormatDesc
#define cudaCreateChannelDesc hipCreateChannelDesc
#define cudaChannelFormatKindUnsigned hipChannelFormatKindUnsigned
#define cudaChannelFormatKindFloat hipChannelFormatKindFloat
#define cudaFilterModePoint hipFilterModePoint
#define cudaFilterModeLinear hipFilterModeLinear
#define cudaTextureDesc hipTextureDesc
#define cudaTextureAddressMode hipTextureAddressMode
#define cudaTextureReadMode hipTextureReadMode
#define cudaAddressModeWrap hipAddressModeWrap
#define cudaAddressModeClamp hipAddressModeClamp
#define cudaAddressModeBorder hipAddressModeBorder
#define cudaReadModeNormalizedFloat hipReadModeNormalizedFloat
#define cudaReadModeElementType hipReadModeElementType
#define cudaResourceDesc hipResourceDesc
#define cudaResourceTypeArray hipResourceTypeArray
#define cudaResourceTypeMipmappedArray hipResourceTypeMipmappedArray

#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventQuery hipEventQuery

#define CUstream hipStream_t
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize

#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize

#endif  // PBRT_UTIL_HIP_ALIASES_H
