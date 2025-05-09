#ifndef CUBEDS_H
#define CUBEDS_H

#include <vector>      // for std::vector
#include <cstdio>      // <stdio.h> works too, but <cstdio> is the C++ header
#include "Vector3f.h"
#include <pbrt/util/spectrum.h>

// Forward‑declare or include the header that defines Vector3f

struct Texel
{
    Vector3f position;
    pbrt::SampledSpectrum transmittance;
};

struct CubeDS
{
    std::vector<Texel> Texels; // OK once <vector> is included
    int resolution;
    float h;
};

#endif // CUBEDS_H
