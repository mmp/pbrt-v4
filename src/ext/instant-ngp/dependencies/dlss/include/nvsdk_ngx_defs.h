/*
* Copyright (c) 2018 NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from NVIDIA Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES.
*/

#ifndef NVSDK_NGX_DEFS_H
#define NVSDK_NGX_DEFS_H
#pragma once

#ifndef __cplusplus
#include <stddef.h> // For size_t
#include <stdbool.h> 
#include <wchar.h> 
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
#if defined(NVSDK_NGX) && defined(NV_WINDOWS)
#define NVSDK_NGX_API extern "C" __declspec(dllexport)
#else
#define NVSDK_NGX_API extern "C"
#endif
#else
#if defined(NVSDK_NGX) && defined(NV_WINDOWS)
#define NVSDK_NGX_API __declspec(dllexport)
#else
#define NVSDK_NGX_API 
#endif
#endif

#ifdef __GNUC__
#define NVSDK_CONV
#else
#define NVSDK_CONV __cdecl
#endif

#define NVSDK_NGX_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
    
//  Version Notes:
//      Version 0x0000014:
//          * Added a logging callback that the app may pass in on init
//          * Added ability for the app to override the logging level
//      Version 0x0000015:
//          * Support multiple GPUs (bug 3270533)
#define NVSDK_NGX_VERSION_API_MACRO 0x0000014  // NGX_VERSION_DOT 1.4.0

typedef struct NVSDK_NGX_FeatureCommonInfo_Internal NVSDK_NGX_FeatureCommonInfo_Internal;

typedef enum NVSDK_NGX_Version { NVSDK_NGX_Version_API = NVSDK_NGX_VERSION_API_MACRO } NVSDK_NGX_Version;

typedef enum NVSDK_NGX_Result
{
    NVSDK_NGX_Result_Success = 0x1,

    NVSDK_NGX_Result_Fail = 0xBAD00000,

    // Feature is not supported on current hardware
    NVSDK_NGX_Result_FAIL_FeatureNotSupported = NVSDK_NGX_Result_Fail | 1,

    // Platform error - for example - check d3d12 debug layer log for more information
    NVSDK_NGX_Result_FAIL_PlatformError = NVSDK_NGX_Result_Fail | 2,

    // Feature with given parameters already exists
    NVSDK_NGX_Result_FAIL_FeatureAlreadyExists = NVSDK_NGX_Result_Fail | 3,

    // Feature with provided handle does not exist
    NVSDK_NGX_Result_FAIL_FeatureNotFound = NVSDK_NGX_Result_Fail | 4,

    // Invalid parameter was provided
    NVSDK_NGX_Result_FAIL_InvalidParameter = NVSDK_NGX_Result_Fail | 5,

    // Provided buffer is too small, please use size provided by NVSDK_NGX_GetScratchBufferSize
    NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall = NVSDK_NGX_Result_Fail | 6,

    // SDK was not initialized properly
    NVSDK_NGX_Result_FAIL_NotInitialized = NVSDK_NGX_Result_Fail | 7,

    //  Unsupported format used for input/output buffers
    NVSDK_NGX_Result_FAIL_UnsupportedInputFormat = NVSDK_NGX_Result_Fail | 8,

    // Feature input/output needs RW access (UAV) (d3d11/d3d12 specific)
    NVSDK_NGX_Result_FAIL_RWFlagMissing = NVSDK_NGX_Result_Fail | 9,

    // Feature was created with specific input but none is provided at evaluation
    NVSDK_NGX_Result_FAIL_MissingInput = NVSDK_NGX_Result_Fail | 10,

    // Feature is not available on the system
    NVSDK_NGX_Result_FAIL_UnableToInitializeFeature = NVSDK_NGX_Result_Fail | 11,

    // NGX system libraries are old and need an update
    NVSDK_NGX_Result_FAIL_OutOfDate = NVSDK_NGX_Result_Fail | 12,

    // Feature requires more GPU memory than it is available on system
    NVSDK_NGX_Result_FAIL_OutOfGPUMemory = NVSDK_NGX_Result_Fail | 13,

    // Format used in input buffer(s) is not supported by feature
    NVSDK_NGX_Result_FAIL_UnsupportedFormat = NVSDK_NGX_Result_Fail | 14,

    // Path provided in InApplicationDataPath cannot be written to
    NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath = NVSDK_NGX_Result_Fail | 15,

    // Unsupported parameter was provided (e.g. specific scaling factor is unsupported)
    NVSDK_NGX_Result_FAIL_UnsupportedParameter = NVSDK_NGX_Result_Fail | 16,

    // The feature or application was denied (contact NVIDIA for further details)
    NVSDK_NGX_Result_FAIL_Denied = NVSDK_NGX_Result_Fail | 17
} NVSDK_NGX_Result;

#define NVSDK_NGX_SUCCEED(value) (((value) & 0xFFF00000) != NVSDK_NGX_Result_Fail)
#define NVSDK_NGX_FAILED(value) (((value) & 0xFFF00000) == NVSDK_NGX_Result_Fail)

typedef enum NVSDK_NGX_Feature
{
    NVSDK_NGX_Feature_Reserved0,

    NVSDK_NGX_Feature_SuperSampling,

    NVSDK_NGX_Feature_InPainting,

    NVSDK_NGX_Feature_ImageSuperResolution,

    NVSDK_NGX_Feature_SlowMotion,

    NVSDK_NGX_Feature_VideoSuperResolution,

    NVSDK_NGX_Feature_Reserved1,

    NVSDK_NGX_Feature_Reserved2,

    NVSDK_NGX_Feature_Reserved3,

    NVSDK_NGX_Feature_ImageSignalProcessing,

    NVSDK_NGX_Feature_DeepResolve,

    NVSDK_NGX_Feature_Reserved4,

    // New features go here
    NVSDK_NGX_Feature_Count,

    // These members are not strictly NGX features, but are 
    // components of the NGX system, and it may sometimes
    // be useful to identify them using the same enum
    NVSDK_NGX_Feature_Reserved_SDK = 32764,

    NVSDK_NGX_Feature_Reserved_Core,

    NVSDK_NGX_Feature_Reserved_Unknown
} NVSDK_NGX_Feature;

//TODO create grayscale format (R32F?)
typedef enum NVSDK_NGX_Buffer_Format
{
    NVSDK_NGX_Buffer_Format_Unknown,
    NVSDK_NGX_Buffer_Format_RGB8UI,
    NVSDK_NGX_Buffer_Format_RGB16F,
    NVSDK_NGX_Buffer_Format_RGB32F,
    NVSDK_NGX_Buffer_Format_RGBA8UI,
    NVSDK_NGX_Buffer_Format_RGBA16F,
    NVSDK_NGX_Buffer_Format_RGBA32F,
} NVSDK_NGX_Buffer_Format;

typedef enum NVSDK_NGX_PerfQuality_Value
{
    NVSDK_NGX_PerfQuality_Value_MaxPerf,
    NVSDK_NGX_PerfQuality_Value_Balanced,
    NVSDK_NGX_PerfQuality_Value_MaxQuality,
    // Extended PerfQuality modes
    NVSDK_NGX_PerfQuality_Value_UltraPerformance,
    NVSDK_NGX_PerfQuality_Value_UltraQuality,
} NVSDK_NGX_PerfQuality_Value;

typedef enum NVSDK_NGX_RTX_Value
{
    NVSDK_NGX_RTX_Value_Off,
    NVSDK_NGX_RTX_Value_On,
} NVSDK_NGX_RTX_Value;

typedef enum NVSDK_NGX_DLSS_Mode
{
    NVSDK_NGX_DLSS_Mode_Off,        // use existing in-engine AA + upscale solution
    NVSDK_NGX_DLSS_Mode_DLSS_DLISP,
    NVSDK_NGX_DLSS_Mode_DLISP_Only, // use existing in-engine AA solution
    NVSDK_NGX_DLSS_Mode_DLSS,       // DLSS will apply AA and upsample at the same time
} NVSDK_NGX_DLSS_Mode;

typedef struct NVSDK_NGX_Handle { unsigned int Id; } NVSDK_NGX_Handle;

typedef enum NSDK_NGX_GPU_Arch
{
    NVSDK_NGX_GPU_Arch_NotSupported = 0,

    // Match NvAPI's NV_GPU_ARCHITECTURE_ID values for GV100 and TU100 for
    // backwards compatibility with snippets built against NvAPI
    NVSDK_NGX_GPU_Arch_Volta        = 0x0140,
    NVSDK_NGX_GPU_Arch_Turing       = 0x0160,

    // Presumably something newer
    NVSDK_NGX_GPU_Arch_Unknown      = 0x7FFFFFF
} NVSDK_NGX_GPU_Arch;

typedef enum NVSDK_NGX_DLSS_Feature_Flags
{
    NVSDK_NGX_DLSS_Feature_Flags_IsInvalid      = 1 << 31,

    NVSDK_NGX_DLSS_Feature_Flags_None           = 0,
    NVSDK_NGX_DLSS_Feature_Flags_IsHDR          = 1 << 0,
    NVSDK_NGX_DLSS_Feature_Flags_MVLowRes       = 1 << 1,
    NVSDK_NGX_DLSS_Feature_Flags_MVJittered     = 1 << 2,
    NVSDK_NGX_DLSS_Feature_Flags_DepthInverted  = 1 << 3,
    NVSDK_NGX_DLSS_Feature_Flags_Reserved_0     = 1 << 4,
    NVSDK_NGX_DLSS_Feature_Flags_DoSharpening   = 1 << 5,
    NVSDK_NGX_DLSS_Feature_Flags_AutoExposure   = 1 << 6,
} NVSDK_NGX_DLSS_Feature_Flags;

typedef enum NVSDK_NGX_ToneMapperType
{
    NVSDK_NGX_TONEMAPPER_STRING = 0,
    NVSDK_NGX_TONEMAPPER_REINHARD,
    NVSDK_NGX_TONEMAPPER_ONEOVERLUMA,
    NVSDK_NGX_TONEMAPPER_ACES,
    NVSDK_NGX_TONEMAPPERTYPE_NUM
} NVSDK_NGX_ToneMapperType;

typedef enum NVSDK_NGX_GBufferType
{
    NVSDK_NGX_GBUFFER_ALBEDO = 0,
    NVSDK_NGX_GBUFFER_ROUGHNESS,
    NVSDK_NGX_GBUFFER_METALLIC,
    NVSDK_NGX_GBUFFER_SPECULAR,
    NVSDK_NGX_GBUFFER_SUBSURFACE,
    NVSDK_NGX_GBUFFER_NORMALS,
    NVSDK_NGX_GBUFFER_SHADINGMODELID,  /* unique identifier for drawn object or how the object is drawn */
    NVSDK_NGX_GBUFFER_MATERIALID, /* unique identifier for material */
    NVSDK_NGX_GBUFFER_SPECULAR_ALBEDO,
    NVSDK_NGX_GBUFFER_INDIRECT_ALBEDO,
    NVSDK_NGX_GBUFFER_SPECULAR_MVEC,
    NVSDK_NGX_GBUFFER_DISOCCL_MASK,
    NVSDK_NGX_GBUFFERTYPE_NUM = 16
} NVSDK_NGX_GBufferType;

typedef struct NVSDK_NGX_Coordinates
{
    unsigned int X;
    unsigned int Y;
} NVSDK_NGX_Coordinates;

typedef struct NVSDK_NGX_Dimensions
{
    unsigned int Width;
    unsigned int Height;
} NVSDK_NGX_Dimensions;

typedef struct NVSDK_NGX_PathListInfo
{
#ifdef NV_WINDOWS
    wchar_t **Path;
#else //NV_WINDOWS
    char **Path;
#endif //NV_WINDOWS
    // Path-list length
    unsigned int Length;
} NVSDK_NGX_PathListInfo;

typedef enum NVSDK_NGX_Logging_Level
{
    NVSDK_NGX_LOGGING_LEVEL_OFF = 0,
    NVSDK_NGX_LOGGING_LEVEL_ON,
    NVSDK_NGX_LOGGING_LEVEL_VERBOSE,
    NVSDK_NGX_LOGGING_LEVEL_NUM
} NVSDK_NGX_Logging_Level;

// A logging callback provided by the app to allow piping log lines back to the app.
// Please take careful note of the signature and calling convention.
// The callback must be able to be called from any thread.
// It must also be fully thread-safe and any number of threads may call into it concurrently. 
// It must fully process message by the time it returns, and there is no guarantee that
// message will still be valid or allocated after it returns.
// message will be a null-terminated string and may contain multibyte characters.
#if defined(__GNUC__) || defined(__clang__)
typedef void NVSDK_CONV(*NVSDK_NGX_AppLogCallback)(const char* message, NVSDK_NGX_Logging_Level loggingLevel, NVSDK_NGX_Feature sourceComponent);
#else
typedef void(NVSDK_CONV* NVSDK_NGX_AppLogCallback)(const char* message, NVSDK_NGX_Logging_Level loggingLevel, NVSDK_NGX_Feature sourceComponent);
#endif

typedef struct NGSDK_NGX_LoggingInfo
{
    // Fields below were introduced in SDK version 0x0000014

    // App-provided logging callback
    NVSDK_NGX_AppLogCallback LoggingCallback;

    // The minimum logging level to use. If this is higher
    // than the logging level otherwise configured, this will override
    // that logging level. Otherwise, that logging level will be used.
    NVSDK_NGX_Logging_Level MinimumLoggingLevel;

    // Whether or not to disable writing log lines to sinks other than the app log callback. This
    // may be useful if the app provides a logging callback. LoggingCallback must be non-null and point
    // to a valid logging callback if this is set to true.
    bool DisableOtherLoggingSinks;

} NGSDK_NGX_LoggingInfo;

typedef struct NVSDK_NGX_FeatureCommonInfo
{
    // List of all paths in descending order of search sequence to locate a feature dll in, other than the default path - application folder.
    NVSDK_NGX_PathListInfo PathListInfo;
    // Used internally by NGX
    NVSDK_NGX_FeatureCommonInfo_Internal* InternalData; // Introduced in SDK version 0x0000013

    // Fields below were introduced in SDK version 0x0000014

    NGSDK_NGX_LoggingInfo LoggingInfo;
} NVSDK_NGX_FeatureCommonInfo;

typedef enum NVSDK_NGX_Resource_VK_Type
{
    NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,
    NVSDK_NGX_RESOURCE_VK_TYPE_VK_BUFFER
} NVSDK_NGX_Resource_VK_Type;

typedef enum NVSDK_NGX_Opt_Level
{
    NVSDK_NGX_OPT_LEVEL_UNDEFINED = 0,
    NVSDK_NGX_OPT_LEVEL_DEBUG = 20,
    NVSDK_NGX_OPT_LEVEL_DEVELOP = 30,
    NVSDK_NGX_OPT_LEVEL_RELEASE = 40
} NVSDK_NGX_Opt_Level;

typedef enum NVSDK_NGX_EngineType
{
    NVSDK_NGX_ENGINE_TYPE_CUSTOM = 0,
    NVSDK_NGX_ENGINE_TYPE_UNREAL,
    NVSDK_NGX_ENGINE_TYPE_UNITY,
    NVSDK_NGX_ENGINE_TYPE_OMNIVERSE,
    NVSDK_NGX_ENGINE_COUNT
} NVSDK_NGX_EngineType;

// Read-only parameters provided by NGX
#define NVSDK_NGX_EParameter_Reserved00                           "#\x00"
#define NVSDK_NGX_EParameter_SuperSampling_Available              "#\x01"
#define NVSDK_NGX_EParameter_InPainting_Available                 "#\x02"
#define NVSDK_NGX_EParameter_ImageSuperResolution_Available       "#\x03"
#define NVSDK_NGX_EParameter_SlowMotion_Available                 "#\x04"
#define NVSDK_NGX_EParameter_VideoSuperResolution_Available       "#\x05"
#define NVSDK_NGX_EParameter_Reserved06                           "#\x06"
#define NVSDK_NGX_EParameter_Reserved07                           "#\x07"
#define NVSDK_NGX_EParameter_Reserved08                           "#\x08"
#define NVSDK_NGX_EParameter_ImageSignalProcessing_Available      "#\x09"
#define NVSDK_NGX_EParameter_ImageSuperResolution_ScaleFactor_2_1 "#\x0a"
#define NVSDK_NGX_EParameter_ImageSuperResolution_ScaleFactor_3_1 "#\x0b"
#define NVSDK_NGX_EParameter_ImageSuperResolution_ScaleFactor_3_2 "#\x0c"
#define NVSDK_NGX_EParameter_ImageSuperResolution_ScaleFactor_4_3 "#\x0d"
#define NVSDK_NGX_EParameter_NumFrames           "#\x0e"
#define NVSDK_NGX_EParameter_Scale               "#\x0f"
#define NVSDK_NGX_EParameter_Width               "#\x10"
#define NVSDK_NGX_EParameter_Height              "#\x11"
#define NVSDK_NGX_EParameter_OutWidth            "#\x12"
#define NVSDK_NGX_EParameter_OutHeight           "#\x13"
#define NVSDK_NGX_EParameter_Sharpness           "#\x14"
#define NVSDK_NGX_EParameter_Scratch             "#\x15"
#define NVSDK_NGX_EParameter_Scratch_SizeInBytes "#\x16"
#define NVSDK_NGX_EParameter_EvaluationNode      "#\x17" // valid since API 0x13 (replaced a deprecated param)
#define NVSDK_NGX_EParameter_Input1              "#\x18"
#define NVSDK_NGX_EParameter_Input1_Format       "#\x19"
#define NVSDK_NGX_EParameter_Input1_SizeInBytes  "#\x1a"
#define NVSDK_NGX_EParameter_Input2              "#\x1b"
#define NVSDK_NGX_EParameter_Input2_Format       "#\x1c"
#define NVSDK_NGX_EParameter_Input2_SizeInBytes  "#\x1d"
#define NVSDK_NGX_EParameter_Color               "#\x1e"
#define NVSDK_NGX_EParameter_Color_Format        "#\x1f"
#define NVSDK_NGX_EParameter_Color_SizeInBytes   "#\x20"
#define NVSDK_NGX_EParameter_Albedo              "#\x21"
#define NVSDK_NGX_EParameter_Output              "#\x22"
#define NVSDK_NGX_EParameter_Output_Format       "#\x23"
#define NVSDK_NGX_EParameter_Output_SizeInBytes  "#\x24"
#define NVSDK_NGX_EParameter_Reset               "#\x25"
#define NVSDK_NGX_EParameter_BlendFactor         "#\x26"
#define NVSDK_NGX_EParameter_MotionVectors       "#\x27"
#define NVSDK_NGX_EParameter_Rect_X              "#\x28"
#define NVSDK_NGX_EParameter_Rect_Y              "#\x29"
#define NVSDK_NGX_EParameter_Rect_W              "#\x2a"
#define NVSDK_NGX_EParameter_Rect_H              "#\x2b"
#define NVSDK_NGX_EParameter_MV_Scale_X          "#\x2c"
#define NVSDK_NGX_EParameter_MV_Scale_Y          "#\x2d"
#define NVSDK_NGX_EParameter_Model               "#\x2e"
#define NVSDK_NGX_EParameter_Format              "#\x2f"
#define NVSDK_NGX_EParameter_SizeInBytes         "#\x30"
#define NVSDK_NGX_EParameter_ResourceAllocCallback      "#\x31"
#define NVSDK_NGX_EParameter_BufferAllocCallback        "#\x32"
#define NVSDK_NGX_EParameter_Tex2DAllocCallback         "#\x33"
#define NVSDK_NGX_EParameter_ResourceReleaseCallback    "#\x34"
#define NVSDK_NGX_EParameter_CreationNodeMask           "#\x35"
#define NVSDK_NGX_EParameter_VisibilityNodeMask         "#\x36"
#define NVSDK_NGX_EParameter_PreviousOutput             "#\x37"
#define NVSDK_NGX_EParameter_MV_Offset_X                 "#\x38"
#define NVSDK_NGX_EParameter_MV_Offset_Y                 "#\x39"
#define NVSDK_NGX_EParameter_Hint_UseFireflySwatter      "#\x3a"
#define NVSDK_NGX_EParameter_Resource_Width              "#\x3b"
#define NVSDK_NGX_EParameter_Resource_Height             "#\x3c"
#define NVSDK_NGX_EParameter_Depth                       "#\x3d"
#define NVSDK_NGX_EParameter_DLSSOptimalSettingsCallback "#\x3e"
#define NVSDK_NGX_EParameter_PerfQualityValue            "#\x3f"
#define NVSDK_NGX_EParameter_RTXValue                    "#\x40"
#define NVSDK_NGX_EParameter_DLSSMode                    "#\x41"
#define NVSDK_NGX_EParameter_DeepResolve_Available       "#\x42"
#define NVSDK_NGX_EParameter_Deprecated_43               "#\x43"
#define NVSDK_NGX_EParameter_OptLevel                    "#\x44"
#define NVSDK_NGX_EParameter_IsDevSnippetBranch          "#\x45"

#define NVSDK_NGX_Parameter_OptLevel "Snippet.OptLevel"
#define NVSDK_NGX_Parameter_IsDevSnippetBranch "Snippet.IsDevBranch"
#define NVSDK_NGX_Parameter_SuperSampling_ScaleFactor  "SuperSampling.ScaleFactor"
#define NVSDK_NGX_Parameter_ImageSignalProcessing_ScaleFactor "ImageSignalProcessing.ScaleFactor"
#define NVSDK_NGX_Parameter_SuperSampling_Available "SuperSampling.Available"
#define NVSDK_NGX_Parameter_InPainting_Available "InPainting.Available"
#define NVSDK_NGX_Parameter_ImageSuperResolution_Available "ImageSuperResolution.Available"
#define NVSDK_NGX_Parameter_SlowMotion_Available "SlowMotion.Available"
#define NVSDK_NGX_Parameter_VideoSuperResolution_Available "VideoSuperResolution.Available"
#define NVSDK_NGX_Parameter_ImageSignalProcessing_Available "ImageSignalProcessing.Available"
#define NVSDK_NGX_Parameter_DeepResolve_Available "DeepResolve.Available"
#define NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver            "SuperSampling.NeedsUpdatedDriver"
#define NVSDK_NGX_Parameter_InPainting_NeedsUpdatedDriver               "InPainting.NeedsUpdatedDriver"
#define NVSDK_NGX_Parameter_ImageSuperResolution_NeedsUpdatedDriver     "ImageSuperResolution.NeedsUpdatedDriver"
#define NVSDK_NGX_Parameter_SlowMotion_NeedsUpdatedDriver               "SlowMotion.NeedsUpdatedDriver"
#define NVSDK_NGX_Parameter_VideoSuperResolution_NeedsUpdatedDriver     "VideoSuperResolution.NeedsUpdatedDriver"
#define NVSDK_NGX_Parameter_ImageSignalProcessing_NeedsUpdatedDriver    "ImageSignalProcessing.NeedsUpdatedDriver"
#define NVSDK_NGX_Parameter_DeepResolve_NeedsUpdatedDriver              "DeepResolve.NeedsUpdatedDriver"
#define NVSDK_NGX_Parameter_FrameInterpolation_NeedsUpdatedDriver       "FrameInterpolation.NeedsUpdatedDriver"
#define NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor         "SuperSampling.MinDriverVersionMajor"
#define NVSDK_NGX_Parameter_InPainting_MinDriverVersionMajor            "InPainting.MinDriverVersionMajor"
#define NVSDK_NGX_Parameter_ImageSuperResolution_MinDriverVersionMajor  "ImageSuperResolution.MinDriverVersionMajor"
#define NVSDK_NGX_Parameter_SlowMotion_MinDriverVersionMajor            "SlowMotion.MinDriverVersionMajor"
#define NVSDK_NGX_Parameter_VideoSuperResolution_MinDriverVersionMajor  "VideoSuperResolution.MinDriverVersionMajor"
#define NVSDK_NGX_Parameter_ImageSignalProcessing_MinDriverVersionMajor "ImageSignalProcessing.MinDriverVersionMajor"
#define NVSDK_NGX_Parameter_DeepResolve_MinDriverVersionMajor           "DeepResolve.MinDriverVersionMajor"
#define NVSDK_NGX_Parameter_FrameInterpolation_MinDriverVersionMajor    "FrameInterpolation.MinDriverVersionMajor"
#define NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor         "SuperSampling.MinDriverVersionMinor"
#define NVSDK_NGX_Parameter_InPainting_MinDriverVersionMinor            "InPainting.MinDriverVersionMinor"
#define NVSDK_NGX_Parameter_ImageSuperResolution_MinDriverVersionMinor  "ImageSuperResolution.MinDriverVersionMinor"
#define NVSDK_NGX_Parameter_SlowMotion_MinDriverVersionMinor            "SlowMotion.MinDriverVersionMinor"
#define NVSDK_NGX_Parameter_VideoSuperResolution_MinDriverVersionMinor  "VideoSuperResolution.MinDriverVersionMinor"
#define NVSDK_NGX_Parameter_ImageSignalProcessing_MinDriverVersionMinor "ImageSignalProcessing.MinDriverVersionMinor"
#define NVSDK_NGX_Parameter_DeepResolve_MinDriverVersionMinor           "DeepResolve.MinDriverVersionMinor"
#define NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult             "SuperSampling.FeatureInitResult"
#define NVSDK_NGX_Parameter_InPainting_FeatureInitResult                "InPainting.FeatureInitResult"
#define NVSDK_NGX_Parameter_ImageSuperResolution_FeatureInitResult      "ImageSuperResolution.FeatureInitResult"
#define NVSDK_NGX_Parameter_SlowMotion_FeatureInitResult                "SlowMotion.FeatureInitResult"
#define NVSDK_NGX_Parameter_VideoSuperResolution_FeatureInitResult      "VideoSuperResolution.FeatureInitResult"
#define NVSDK_NGX_Parameter_ImageSignalProcessing_FeatureInitResult     "ImageSignalProcessing.FeatureInitResult"
#define NVSDK_NGX_Parameter_DeepResolve_FeatureInitResult               "DeepResolve.FeatureInitResult"
#define NVSDK_NGX_Parameter_FrameInterpolation_FeatureInitResult        "FrameInterpolation.FeatureInitResult"
#define NVSDK_NGX_Parameter_ImageSuperResolution_ScaleFactor_2_1 "ImageSuperResolution.ScaleFactor.2.1"
#define NVSDK_NGX_Parameter_ImageSuperResolution_ScaleFactor_3_1 "ImageSuperResolution.ScaleFactor.3.1"
#define NVSDK_NGX_Parameter_ImageSuperResolution_ScaleFactor_3_2 "ImageSuperResolution.ScaleFactor.3.2"
#define NVSDK_NGX_Parameter_ImageSuperResolution_ScaleFactor_4_3 "ImageSuperResolution.ScaleFactor.4.3"
#define NVSDK_NGX_Parameter_NumFrames "NumFrames"
#define NVSDK_NGX_Parameter_Scale "Scale"
#define NVSDK_NGX_Parameter_Width "Width"
#define NVSDK_NGX_Parameter_Height "Height"
#define NVSDK_NGX_Parameter_OutWidth "OutWidth"
#define NVSDK_NGX_Parameter_OutHeight "OutHeight"
#define NVSDK_NGX_Parameter_Sharpness "Sharpness"
#define NVSDK_NGX_Parameter_Scratch "Scratch"
#define NVSDK_NGX_Parameter_Scratch_SizeInBytes "Scratch.SizeInBytes"
#define NVSDK_NGX_Parameter_Input1 "Input1"
#define NVSDK_NGX_Parameter_Input1_Format "Input1.Format"
#define NVSDK_NGX_Parameter_Input1_SizeInBytes "Input1.SizeInBytes"
#define NVSDK_NGX_Parameter_Input2 "Input2"
#define NVSDK_NGX_Parameter_Input2_Format "Input2.Format"
#define NVSDK_NGX_Parameter_Input2_SizeInBytes "Input2.SizeInBytes"
#define NVSDK_NGX_Parameter_Color "Color"
#define NVSDK_NGX_Parameter_Color_Format "Color.Format"
#define NVSDK_NGX_Parameter_Color_SizeInBytes "Color.SizeInBytes"
#define NVSDK_NGX_Parameter_FI_Color1 "Color1"
#define NVSDK_NGX_Parameter_FI_Color2 "Color2"
#define NVSDK_NGX_Parameter_Albedo "Albedo"
#define NVSDK_NGX_Parameter_Output "Output"
#define NVSDK_NGX_Parameter_Output_SizeInBytes "Output.SizeInBytes"
#define NVSDK_NGX_Parameter_FI_Output1 "Output1"
#define NVSDK_NGX_Parameter_FI_Output2 "Output2"
#define NVSDK_NGX_Parameter_FI_Output3 "Output3"
#define NVSDK_NGX_Parameter_Reset "Reset"
#define NVSDK_NGX_Parameter_BlendFactor "BlendFactor"
#define NVSDK_NGX_Parameter_MotionVectors "MotionVectors"
#define NVSDK_NGX_Parameter_FI_MotionVectors1 "MotionVectors1"
#define NVSDK_NGX_Parameter_FI_MotionVectors2 "MotionVectors2"
#define NVSDK_NGX_Parameter_Rect_X "Rect.X"
#define NVSDK_NGX_Parameter_Rect_Y "Rect.Y"
#define NVSDK_NGX_Parameter_Rect_W "Rect.W"
#define NVSDK_NGX_Parameter_Rect_H "Rect.H"
#define NVSDK_NGX_Parameter_MV_Scale_X "MV.Scale.X"
#define NVSDK_NGX_Parameter_MV_Scale_Y "MV.Scale.Y"
#define NVSDK_NGX_Parameter_Model "Model"
#define NVSDK_NGX_Parameter_Format "Format"
#define NVSDK_NGX_Parameter_SizeInBytes "SizeInBytes"
#define NVSDK_NGX_Parameter_ResourceAllocCallback      "ResourceAllocCallback"
#define NVSDK_NGX_Parameter_BufferAllocCallback        "BufferAllocCallback"
#define NVSDK_NGX_Parameter_Tex2DAllocCallback         "Tex2DAllocCallback"
#define NVSDK_NGX_Parameter_ResourceReleaseCallback    "ResourceReleaseCallback"
#define NVSDK_NGX_Parameter_CreationNodeMask           "CreationNodeMask"
#define NVSDK_NGX_Parameter_VisibilityNodeMask         "VisibilityNodeMask"
#define NVSDK_NGX_Parameter_MV_Offset_X "MV.Offset.X"
#define NVSDK_NGX_Parameter_MV_Offset_Y "MV.Offset.Y"
#define NVSDK_NGX_Parameter_Hint_UseFireflySwatter "Hint.UseFireflySwatter"
#define NVSDK_NGX_Parameter_Resource_Width "ResourceWidth"
#define NVSDK_NGX_Parameter_Resource_Height "ResourceHeight"
#define NVSDK_NGX_Parameter_Resource_OutWidth "ResourceOutWidth"
#define NVSDK_NGX_Parameter_Resource_OutHeight "ResourceOutHeight"
#define NVSDK_NGX_Parameter_Depth "Depth"
#define NVSDK_NGX_Parameter_FI_Depth1 "Depth1"
#define NVSDK_NGX_Parameter_FI_Depth2 "Depth2"
#define NVSDK_NGX_Parameter_DLSSOptimalSettingsCallback    "DLSSOptimalSettingsCallback"
#define NVSDK_NGX_Parameter_DLSSGetStatsCallback    "DLSSGetStatsCallback"
#define NVSDK_NGX_Parameter_PerfQualityValue    "PerfQualityValue"
#define NVSDK_NGX_Parameter_RTXValue    "RTXValue"
#define NVSDK_NGX_Parameter_DLSSMode    "DLSSMode"
#define NVSDK_NGX_Parameter_FI_Mode     "FIMode"
#define NVSDK_NGX_Parameter_FI_OF_Preset    "FIOFPreset"
#define NVSDK_NGX_Parameter_FI_OF_GridSize  "FIOFGridSize"
#define NVSDK_NGX_Parameter_Jitter_Offset_X     "Jitter.Offset.X"
#define NVSDK_NGX_Parameter_Jitter_Offset_Y     "Jitter.Offset.Y"
#define NVSDK_NGX_Parameter_Denoise "Denoise"
#define NVSDK_NGX_Parameter_TransparencyMask "TransparencyMask"
#define NVSDK_NGX_Parameter_ExposureTexture   "ExposureTexture" // a 1x1 texture containing the final exposure scale
#define NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags "DLSS.Feature.Create.Flags"
#define NVSDK_NGX_Parameter_DLSS_Checkerboard_Jitter_Hack "DLSS.Checkerboard.Jitter.Hack"
#define NVSDK_NGX_Parameter_GBuffer_Normals "GBuffer.Normals"
#define NVSDK_NGX_Parameter_GBuffer_Albedo "GBuffer.Albedo"
#define NVSDK_NGX_Parameter_GBuffer_Roughness "GBuffer.Roughness"
#define NVSDK_NGX_Parameter_GBuffer_DiffuseAlbedo "GBuffer.DiffuseAlbedo"
#define NVSDK_NGX_Parameter_GBuffer_SpecularAlbedo "GBuffer.SpecularAlbedo"
#define NVSDK_NGX_Parameter_GBuffer_IndirectAlbedo "GBuffer.IndirectAlbedo"
#define NVSDK_NGX_Parameter_GBuffer_SpecularMvec "GBuffer.SpecularMvec"
#define NVSDK_NGX_Parameter_GBuffer_DisocclusionMask "GBuffer.DisocclusionMask"
#define NVSDK_NGX_Parameter_GBuffer_Metallic "GBuffer.Metallic"
#define NVSDK_NGX_Parameter_GBuffer_Specular "GBuffer.Specular"
#define NVSDK_NGX_Parameter_GBuffer_Subsurface "GBuffer.Subsurface"
#define NVSDK_NGX_Parameter_GBuffer_Normals "GBuffer.Normals"
#define NVSDK_NGX_Parameter_GBuffer_ShadingModelId "GBuffer.ShadingModelId"
#define NVSDK_NGX_Parameter_GBuffer_MaterialId "GBuffer.MaterialId"
#define NVSDK_NGX_Parameter_GBuffer_Atrrib_8 "GBuffer.Attrib.8"
#define NVSDK_NGX_Parameter_GBuffer_Atrrib_9 "GBuffer.Attrib.9"
#define NVSDK_NGX_Parameter_GBuffer_Atrrib_10 "GBuffer.Attrib.10"
#define NVSDK_NGX_Parameter_GBuffer_Atrrib_11 "GBuffer.Attrib.11"
#define NVSDK_NGX_Parameter_GBuffer_Atrrib_12 "GBuffer.Attrib.12"
#define NVSDK_NGX_Parameter_GBuffer_Atrrib_13 "GBuffer.Attrib.13"
#define NVSDK_NGX_Parameter_GBuffer_Atrrib_14 "GBuffer.Attrib.14"
#define NVSDK_NGX_Parameter_GBuffer_Atrrib_15 "GBuffer.Attrib.15"
#define NVSDK_NGX_Parameter_TonemapperType "TonemapperType"
#define NVSDK_NGX_Parameter_FreeMemOnReleaseFeature "FreeMemOnReleaseFeature"
#define NVSDK_NGX_Parameter_MotionVectors3D "MotionVectors3D"
#define NVSDK_NGX_Parameter_IsParticleMask "IsParticleMask"
#define NVSDK_NGX_Parameter_AnimatedTextureMask "AnimatedTextureMask"
#define NVSDK_NGX_Parameter_DepthHighRes "DepthHighRes"
#define NVSDK_NGX_Parameter_Position_ViewSpace "Position.ViewSpace"
#define NVSDK_NGX_Parameter_FrameTimeDeltaInMsec "FrameTimeDeltaInMsec"
#define NVSDK_NGX_Parameter_RayTracingHitDistance "RayTracingHitDistance"
#define NVSDK_NGX_Parameter_MotionVectorsReflection "MotionVectorsReflection"
#define NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects "DLSS.Enable.Output.Subrects"
#define NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X "DLSS.Input.Color.Subrect.Base.X"
#define NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y "DLSS.Input.Color.Subrect.Base.Y"
#define NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X "DLSS.Input.Depth.Subrect.Base.X"
#define NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y "DLSS.Input.Depth.Subrect.Base.Y"
#define NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X "DLSS.Input.MV.Subrect.Base.X"
#define NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y "DLSS.Input.MV.Subrect.Base.Y"
#define NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_X "DLSS.Input.Translucency.Subrect.Base.X"
#define NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_Y "DLSS.Input.Translucency.Subrect.Base.Y"
#define NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X "DLSS.Output.Subrect.Base.X"
#define NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y "DLSS.Output.Subrect.Base.Y"
#define NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width  "DLSS.Render.Subrect.Dimensions.Width"
#define NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height "DLSS.Render.Subrect.Dimensions.Height"
#define NVSDK_NGX_Parameter_DLSS_Pre_Exposure "DLSS.Pre.Exposure"
#define NVSDK_NGX_Parameter_DLSS_Exposure_Scale "DLSS.Exposure.Scale"
#define NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask          "DLSS.Input.Bias.Current.Color.Mask"
#define NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_X "DLSS.Input.Bias.Current.Color.Subrect.Base.X"
#define NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_Y "DLSS.Input.Bias.Current.Color.Subrect.Base.Y"
#define NVSDK_NGX_Parameter_DLSS_Indicator_Invert_Y_Axis          "DLSS.Indicator.Invert.Y.Axis"
#define NVSDK_NGX_Parameter_DLSS_Indicator_Invert_X_Axis          "DLSS.Indicator.Invert.X.Axis"

#define NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Width     "DLSS.Get.Dynamic.Max.Render.Width"
#define NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Height    "DLSS.Get.Dynamic.Max.Render.Height"
#define NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Width     "DLSS.Get.Dynamic.Min.Render.Width"
#define NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Height    "DLSS.Get.Dynamic.Min.Render.Height"

#ifdef __cplusplus
} // extern "C"
#endif

#endif
