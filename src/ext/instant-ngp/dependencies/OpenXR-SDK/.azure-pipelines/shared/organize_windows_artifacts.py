#!/usr/bin/env python3
# Copyright (c) 2019-2025 The Khronos Group Inc.
# SPDX-License-Identifier: Apache-2.0

from itertools import product
from pathlib import Path
import shutil
import sys

from shared import PLATFORMS, TRUE_FALSE, VS_VERSION, make_win_artifact_name

CWD = Path.cwd()


def move(src, dest):

    print(str(src), '->', str(dest))
    src.replace(dest)


if __name__ == "__main__":

    configs = {}
    workspace = Path(sys.argv[1])
    outbase = Path(sys.argv[2])

    common_copied = False

    for platform, uwp in product(PLATFORMS, TRUE_FALSE):
        # ARM/ARM64 is only built for the UWP platform.
        if not uwp and (platform.lower() == 'arm' or platform.lower() == 'arm64'):
            continue

        platform_dirname = '{}{}'.format(platform,
                                         '_uwp' if uwp else '')

        name = make_win_artifact_name(platform, uwp)

        artifact = workspace / name

        if not common_copied:
            # Start by copying the full tree over.
            shutil.copytree(artifact, outbase, dirs_exist_ok=True)
            common_copied = True
            continue

        # lib files
        shutil.copytree(artifact / platform_dirname, outbase / platform_dirname, dirs_exist_ok=True)
