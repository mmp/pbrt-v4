# DLSS
Public repo for NVIDIA RTX DLSS SDK. 
The DLSS Sample app is included only in the releases. 

## NVIDIA Image Scaling SDK
The NVIDIA Image Scaling SDK provides a single spatial scaling and sharpening algorithm for cross-platform support. It contains compute shaders that can be integrated with DX11, DX12, and Vulkan. For more information visit https://github.com/NVIDIAGameWorks/NVIDIAImageScaling.

To get the NVIDIA Image Scaling SDK submodule use the following command
```
$ git clone --recurse-submodules https://github.com/NVIDIA/DLSS.git
```
or if you already have clonned the DLSS repository and didn't use --recurse-submodules
```
git submodule update --init --recursive
```