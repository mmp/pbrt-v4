#!/usr/bin/env python3
# Copyright (c) 2019-2025 The Khronos Group Inc.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

from shared import (PLATFORMS, TRUE_FALSE, VS_VERSION, make_win_artifact_name,
                    output_json)

if __name__ == "__main__":

    configs = {}
    for  platform, debug, uwp in product(PLATFORMS, (False,), TRUE_FALSE):
        # No need to support ARM/ARM64 except for UWP.
        if not uwp and (platform.lower() == 'arm' or platform.lower() == 'arm64'):
            continue

        label = [platform]
        config = []
        generator = VS_VERSION
        config.append('-A ' + platform)
        config.append('-DDYNAMIC_LOADER=ON')
        if debug:
            label.append('debug')
        if uwp:
            label.append('UWP')
            config.append('-DCMAKE_SYSTEM_NAME=WindowsStore -DCMAKE_SYSTEM_VERSION=10.0')
        name = '_'.join(label)
        configs[name] = {
            'generator': generator,
            'buildType': 'Debug' if debug else 'RelWithDebInfo',
            'cmakeArgs': ' '.join(config)
        }
        if not debug:
            configs[name]['artifactName'] = make_win_artifact_name(
                platform, uwp)

    output_json(configs)
