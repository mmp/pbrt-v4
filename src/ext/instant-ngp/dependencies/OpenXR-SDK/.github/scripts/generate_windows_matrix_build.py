#!/usr/bin/env python3
# Copyright (c) 2019-2025 The Khronos Group Inc.
# SPDX-License-Identifier: Apache-2.0

from itertools import product
import sys

from shared import BUILD_CONFIGS, output_json

if __name__ == "__main__":

    matrix = {"include": [{"preset": c.preset()} for c in BUILD_CONFIGS]}
    var_name = None
    if len(sys.argv) >= 2:
        var_name = sys.argv[1]
    output_json(matrix, variable_name=var_name)
