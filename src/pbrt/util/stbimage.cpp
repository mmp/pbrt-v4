// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/check.h>

#define STBI_NO_PNG
// too old school
#define STBI_NO_PIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ASSERT CHECK
#include <stb/stb_image.h>
