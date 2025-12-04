# Copyright (c) 2019-2025 The Khronos Group Inc.
# SPDX-License-Identifier: Apache-2.0
param(
    [Parameter(Mandatory = $true, HelpMessage = "Path to unzipped openxr_loader_windows OpenXR-SDK release asset")]
    $SDKRelease,
    [Parameter(Mandatory = $true, HelpMessage = "Path to specification Makefile. Needed to extract the version")]
    $SpecMakefile,
    [Parameter(Mandatory = $true, HelpMessage = "Path create staged nuget directory layout")]
    $NugetStaging)

$ErrorActionPreference = "Stop"

if (-Not (Test-Path $SDKRelease)) {
    Throw "SDK Release folder not found: $SDKRelease"
}
if (-Not (Test-Path $SpecMakefile)) {
    Throw "Specification makefile not found: $SpecMakefile"
}

$NugetTemplate = Join-Path $PSScriptRoot "NugetTemplate"

if (Test-Path $NugetStaging) {
    Remove-Item $NugetStaging -Recurse
}

#
# Extract version from Specification makefile
#
$VersionMatch = Select-String -Path $SpecMakefile -Pattern "^SPECREVISION\s*=\s*(.+)"
$SDKVersion = $VersionMatch.Matches[0].Groups[1]

#
# Start off using the NuGet template.
#
Write-Output "Copy-Item $NugetTemplate $NugetStaging -Recurse"
Copy-Item $NugetTemplate $NugetStaging -Recurse

#
# Update the NuSpec
#
$NuSpecPath = Resolve-Path (Join-Path $NugetStaging "OpenXR.Loader.nuspec")
$xml = [xml](Get-Content $NuSpecPath)
$nsm = New-Object Xml.XmlNamespaceManager($xml.NameTable)
$nsm.AddNamespace("ng", "http://schemas.microsoft.com/packaging/2010/07/nuspec.xsd")
$xml.SelectSingleNode("/ng:package/ng:metadata/ng:version", $nsm).InnerText = $SDKVersion
$xml.Save($NuSpecPath)

#
# Copy in the headers from the SDK release.
#
Copy-Item (Join-Path $SDKRelease "include") (Join-Path $NugetStaging "include") -Recurse

#
# Copy in the binaries from the SDK release.
#
function CopyLoader($Platform) {
    $PlatformSDKPath = Join-Path $SDKRelease "$Platform"
    if (Test-Path $PlatformSDKPath) {
        $NuGetPlatformPath = Join-Path $NugetStaging "native/$Platform/release"

        $NugetLibPath = Join-Path $NuGetPlatformPath "lib"
        New-Item $NugetLibPath -ItemType "directory" -Force
        Copy-Item (Join-Path $PlatformSDKPath "lib/openxr_loader.lib") $NugetLibPath

        $NugetBinPath = Join-Path $NuGetPlatformPath "bin"
        New-Item $NugetBinPath -ItemType "directory" -Force
        Copy-Item (Join-Path $PlatformSDKPath "bin/openxr_loader.dll") $NugetBinPath
    }
}

# Currently there are no non-UWP ARM/ARM64 binaries available from the SDK release.
CopyLoader "x64"
CopyLoader "Win32"
CopyLoader "x64_uwp"
CopyLoader "Win32_uwp"
CopyLoader "arm64_uwp"
CopyLoader "arm_uwp"
