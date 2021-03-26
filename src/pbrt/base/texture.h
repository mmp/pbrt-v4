// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_TEXTURE_H
#define PBRT_BASE_TEXTURE_H

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

struct TextureEvalContext;

class FloatConstantTexture;
class FloatBilerpTexture;
class FloatCheckerboardTexture;
class FloatDotsTexture;
class FBmTexture;
class GPUFloatImageTexture;
class FloatImageTexture;
class FloatMixTexture;
class FloatPtexTexture;
class GPUFloatPtexTexture;
class FloatScaledTexture;
class WindyTexture;
class WrinkledTexture;

// FloatTexture Definition
class FloatTexture
    : public TaggedPointer<  // FloatTextures
          FloatImageTexture, GPUFloatImageTexture, FloatMixTexture, FloatScaledTexture,
          FloatConstantTexture, FloatBilerpTexture, FloatCheckerboardTexture,
          FloatDotsTexture, FBmTexture, FloatPtexTexture, GPUFloatPtexTexture,
          WindyTexture, WrinkledTexture

          > {
  public:
    // FloatTexture Interface
    using TaggedPointer::TaggedPointer;

    static FloatTexture Create(const std::string &name,
                               const Transform &renderFromTexture,
                               const TextureParameterDictionary &parameters,
                               const FileLoc *loc, Allocator alloc, bool gpu);

    std::string ToString() const;

    PBRT_CPU_GPU inline Float Evaluate(TextureEvalContext ctx) const;
};

class RGBConstantTexture;
class RGBReflectanceConstantTexture;
class SpectrumConstantTexture;
class SpectrumBilerpTexture;
class SpectrumCheckerboardTexture;
class SpectrumImageTexture;
class GPUSpectrumImageTexture;
class MarbleTexture;
class SpectrumMixTexture;
class SpectrumDotsTexture;
class SpectrumPtexTexture;
class GPUSpectrumPtexTexture;
class SpectrumScaledTexture;

// SpectrumTexture Definition
class SpectrumTexture
    : public TaggedPointer<  // SpectrumTextures
          SpectrumImageTexture, GPUSpectrumImageTexture, SpectrumMixTexture,
          SpectrumScaledTexture, SpectrumConstantTexture, SpectrumBilerpTexture,
          SpectrumCheckerboardTexture, MarbleTexture, SpectrumDotsTexture,
          SpectrumPtexTexture, GPUSpectrumPtexTexture

          > {
  public:
    // SpectrumTexture Interface
    using TaggedPointer::TaggedPointer;

    static SpectrumTexture Create(const std::string &name,
                                  const Transform &renderFromTexture,
                                  const TextureParameterDictionary &parameters,
                                  SpectrumType spectrumType, const FileLoc *loc,
                                  Allocator alloc, bool gpu);

    std::string ToString() const;

    PBRT_CPU_GPU inline SampledSpectrum Evaluate(TextureEvalContext ctx,
                                                 SampledWavelengths lambda) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_TEXTURE_H
