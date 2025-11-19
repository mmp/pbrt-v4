#!/usr/bin/env python3
# Copyright (c) 2019-2025 The Khronos Group Inc.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

from shared import PLATFORMS, TRUE_FALSE, VS_VERSION, make_win_artifact_name

if __name__ == "__main__":

    for platform, uwp in product(PLATFORMS, TRUE_FALSE):
        print(make_win_artifact_name(platform, uwp))
