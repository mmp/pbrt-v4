// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/bssrdf.h>

#include <pbrt/media.h>
#include <pbrt/shapes.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>

#include <cmath>

namespace pbrt {

std::string TabulatedBSSRDF::ToString() const {
    return StringPrintf(
        "[ TabulatedBSSRDF po: %s eta: %f ns: %s sigma_t: %s rho: %s table: %s ]", po,
        eta, ns, sigma_t, rho, *table);
}

// BSSRDF Function Definitions
Float BeamDiffusionMS(Float sigma_s, Float sigma_a, Float g, Float eta, Float r) {
    const int nSamples = 100;
    Float Ed = 0;
    // Precompute information for dipole integrand
    // Compute reduced scattering coefficients $\sigmaps, \sigmapt$ and albedo $\rhop$
    Float sigmap_s = sigma_s * (1 - g);
    Float sigmap_t = sigma_a + sigmap_s;
    Float rhop = sigmap_s / sigmap_t;

    // Compute non-classical diffusion coefficient $D_\roman{G}$ using Equation
    // $(\ref{eq:diffusion-coefficient-grosjean})$
    Float D_g = (2 * sigma_a + sigmap_s) / (3 * sigmap_t * sigmap_t);

    // Compute effective transport coefficient $\sigmatr$ based on $D_\roman{G}$
    Float sigma_tr = SafeSqrt(sigma_a / D_g);

    // Determine linear extrapolation distance $\depthextrapolation$ using Equation
    // $(\ref{eq:dipole-boundary-condition})$
    Float fm1 = FresnelMoment1(eta), fm2 = FresnelMoment2(eta);
    Float ze = -2 * D_g * (1 + 3 * fm2) / (1 - 2 * fm1);

    // Determine exitance scale factors using Equations $(\ref{eq:kp-exitance-phi})$ and
    // $(\ref{eq:kp-exitance-e})$
    Float cPhi = 0.25f * (1 - 2 * fm1), cE = 0.5f * (1 - 3 * fm2);

    for (int i = 0; i < nSamples; ++i) {
        // Sample real point source depth $\depthreal$
        Float zr = SampleExponential((i + 0.5f) / nSamples, sigmap_t);

        // Evaluate dipole integrand $E_{\roman{d}}$ at $\depthreal$ and add to _Ed_
        Float zv = -zr + 2 * ze;
        Float dr = std::sqrt(Sqr(r) + Sqr(zr)), dv = std::sqrt(Sqr(r) + Sqr(zv));
        // Compute dipole fluence rate $\dipole(r)$ using Equation
        // $(\ref{eq:diffusion-dipole})$
        Float phiD =
            Inv4Pi / D_g * (FastExp(-sigma_tr * dr) / dr - FastExp(-sigma_tr * dv) / dv);

        // Compute dipole vector irradiance $-\N{}\cdot\dipoleE(r)$ using Equation
        // $(\ref{eq:diffusion-dipole-vector-irradiance-normal})$
        Float EDn =
            Inv4Pi * (zr * (1 + sigma_tr * dr) * FastExp(-sigma_tr * dr) / (Pow<3>(dr)) -
                      zv * (1 + sigma_tr * dv) * FastExp(-sigma_tr * dv) / (Pow<3>(dv)));

        // Add contribution from dipole for depth $\depthreal$ to _Ed_
        Float E = phiD * cPhi + EDn * cE;
        Float kappa = 1 - FastExp(-2 * sigmap_t * (dr + zr));
        Ed += kappa * rhop * rhop * E;
    }
    return Ed / nSamples;
}

Float BeamDiffusionSS(Float sigma_s, Float sigma_a, Float g, Float eta, Float r) {
    // Compute material parameters and minimum $t$ below the critical angle
    Float sigma_t = sigma_a + sigma_s, rho = sigma_s / sigma_t;
    Float tCrit = r * SafeSqrt(Sqr(eta) - 1);

    Float Ess = 0;
    const int nSamples = 100;
    for (int i = 0; i < nSamples; ++i) {
        // Evaluate single-scattering integrand and add to _Ess_
        Float ti = tCrit + SampleExponential((i + 0.5f) / nSamples, sigma_t);
        // Determine length $d$ of connecting segment and $\cos\theta_\roman{o}$
        Float d = std::sqrt(Sqr(r) + Sqr(ti));
        Float cosTheta_o = ti / d;

        // Add contribution of single scattering at depth $t$
        Ess += rho * FastExp(-sigma_t * (d + tCrit)) / Sqr(d) *
               HenyeyGreenstein(cosTheta_o, g) * (1 - FrDielectric(-cosTheta_o, eta)) *
               std::abs(cosTheta_o);
    }
    return Ess / nSamples;
}

void ComputeBeamDiffusionBSSRDF(Float g, Float eta, BSSRDFTable *t) {
    // Choose radius values of the diffusion profile discretization
    t->radiusSamples[0] = 0;
    t->radiusSamples[1] = 2.5e-3f;
    for (int i = 2; i < t->radiusSamples.size(); ++i)
        t->radiusSamples[i] = t->radiusSamples[i - 1] * 1.2f;

    // Choose albedo values of the diffusion profile discretization
    for (int i = 0; i < t->rhoSamples.size(); ++i)
        t->rhoSamples[i] =
            (1 - FastExp(-8 * i / (Float)(t->rhoSamples.size() - 1))) / (1 - FastExp(-8));

    ParallelFor(0, t->rhoSamples.size(), [&](int i) {
        // Compute the diffusion profile for the _i_th albedo sample
        // Compute scattering profile for chosen albedo $\rho$
        size_t nSamples = t->radiusSamples.size();
        for (int j = 0; j < nSamples; ++j) {
            Float rho = t->rhoSamples[i], r = t->radiusSamples[j];
            t->profile[i * nSamples + j] = 2 * Pi * r *
                                           (BeamDiffusionSS(rho, 1 - rho, g, eta, r) +
                                            BeamDiffusionMS(rho, 1 - rho, g, eta, r));
        }

        // Compute effective albedo $\rho_{\roman{eff}}$ and CDF for importance sampling
        t->rhoEff[i] = IntegrateCatmullRom(
            t->radiusSamples,
            pstd::span<const Float>(&t->profile[i * nSamples], nSamples),
            pstd::span<Float>(&t->profileCDF[i * nSamples], nSamples));
    });
}

// BSSRDFTable Method Definitions
BSSRDFTable::BSSRDFTable(int nRhoSamples, int nRadiusSamples, Allocator alloc)
    : rhoSamples(nRhoSamples, alloc),
      radiusSamples(nRadiusSamples, alloc),
      profile(nRadiusSamples * nRhoSamples, alloc),
      rhoEff(nRhoSamples, alloc),
      profileCDF(nRadiusSamples * nRhoSamples, alloc) {}

std::string BSSRDFTable::ToString() const {
    return StringPrintf("[ BSSRDFTable rhoSamples: %s radiusSamples: %s profile: %s "
                        "rhoEff: %s profileCDF: %s ]",
                        rhoSamples, radiusSamples, profile, rhoEff, profileCDF);
}

}  // namespace pbrt
