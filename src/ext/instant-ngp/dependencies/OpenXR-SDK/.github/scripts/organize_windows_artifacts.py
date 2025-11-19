#!/usr/bin/env python3
# Copyright (c) 2019-2025 The Khronos Group Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import shutil
import sys

from shared import BUILD_CONFIGS

CWD = Path.cwd()

def move(src, dest):

    print(str(src), "->", str(dest))
    src.replace(dest)


if __name__ == "__main__":

    configs = {}
    workspace = Path(sys.argv[1])
    outbase = Path(sys.argv[2])

    common_copied = False

    for config in BUILD_CONFIGS:
        platform_dirname = config.platform_dirname()

        name = config.win_artifact_name()

        artifact = workspace / "artifacts" / name

        if not common_copied:
            # Start by copying the full tree over.
            shutil.copytree(artifact, outbase, dirs_exist_ok=True)
            common_copied = True
            continue

        # lib files
        shutil.copytree(
            artifact / platform_dirname, outbase / platform_dirname, dirs_exist_ok=True
        )
