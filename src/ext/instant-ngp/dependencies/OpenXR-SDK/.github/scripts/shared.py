# Copyright (c) 2019-2025 The Khronos Group Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import os
from dataclasses import dataclass
from itertools import product

VS_VERSION = "Visual Studio 17 2022"

PLATFORMS = ("Win32", "x64", "ARM64")

TRUE_FALSE = (True, False)


@dataclass
class BuildConfig:
    arch: str
    uwp: bool

    def should_skip(self) -> bool:
        # ARM/ARM64 is only built for the UWP platform.
        return "ARM" in self.arch and not self.uwp

        # can switch to just doing x64 for speed of testing
        # return self.arch != "x64"

    def has_cts_build(self) -> bool:
        # No UWP CTS right now
        return not self.uwp and not self.should_skip()

    def preset(self) -> str:
        if self.uwp:
            return f"{self.arch.lower()}_uwp"

        return self.arch.lower()

    def win_artifact_name(self) -> str:
        return f"loader_{self.preset()}"

    def win_cts_artifact_name(self) -> str:
        return f"loader_{self.preset()}"

    def platform_dirname(self) -> str:
        if self.uwp:
            return f"{self.arch}_uwp"
        return self.arch


_UNFILTERED_BUILD_CONFIGS = [
    BuildConfig(arch, uwp) for arch, uwp in product(PLATFORMS, TRUE_FALSE)
]

BUILD_CONFIGS = [c for c in _UNFILTERED_BUILD_CONFIGS if not c.should_skip()]

CTS_BUILD_CONFIGS = [c for c in _UNFILTERED_BUILD_CONFIGS if c.has_cts_build()]


def output_json(data, variable_name=None):
    if variable_name:
        envfile = os.getenv("GITHUB_ENV")
        assert envfile
        with open(envfile, "w", encoding="utf-8") as fp:
            fp.write(f"{variable_name}={json.dumps(data)}")
    else:
        print(json.dumps(data, indent=4))
