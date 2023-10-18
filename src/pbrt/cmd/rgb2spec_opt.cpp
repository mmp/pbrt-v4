// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#if defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define strcasecmp _stricmp
#endif

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

/**
 * This file contains:
 *
 * 1. CIE 1931 curves at sampled at 5nm intervals
 *
 * 2. CIE D65 and D50 spectra sampled at 5nm intervals.
 *    Both are normalized to have unit luminance.
 *
 * 3. XYZ <-> sRGB conversion matrices
 *    XYZ <-> ProPhoto RGB conversion matrices
 *
 * 4. A convenience function "cie_interp" to access the discretized
 *    data at arbitrary wavelengths (with linear interpolation)
}
 */
#define CIE_LAMBDA_MIN 360.0
#define CIE_LAMBDA_MAX 830.0
#define CIE_SAMPLES 95

const double cie_x[CIE_SAMPLES] = {
    0.000129900000, 0.000232100000, 0.000414900000, 0.000741600000, 0.001368000000,
    0.002236000000, 0.004243000000, 0.007650000000, 0.014310000000, 0.023190000000,
    0.043510000000, 0.077630000000, 0.134380000000, 0.214770000000, 0.283900000000,
    0.328500000000, 0.348280000000, 0.348060000000, 0.336200000000, 0.318700000000,
    0.290800000000, 0.251100000000, 0.195360000000, 0.142100000000, 0.095640000000,
    0.057950010000, 0.032010000000, 0.014700000000, 0.004900000000, 0.002400000000,
    0.009300000000, 0.029100000000, 0.063270000000, 0.109600000000, 0.165500000000,
    0.225749900000, 0.290400000000, 0.359700000000, 0.433449900000, 0.512050100000,
    0.594500000000, 0.678400000000, 0.762100000000, 0.842500000000, 0.916300000000,
    0.978600000000, 1.026300000000, 1.056700000000, 1.062200000000, 1.045600000000,
    1.002600000000, 0.938400000000, 0.854449900000, 0.751400000000, 0.642400000000,
    0.541900000000, 0.447900000000, 0.360800000000, 0.283500000000, 0.218700000000,
    0.164900000000, 0.121200000000, 0.087400000000, 0.063600000000, 0.046770000000,
    0.032900000000, 0.022700000000, 0.015840000000, 0.011359160000, 0.008110916000,
    0.005790346000, 0.004109457000, 0.002899327000, 0.002049190000, 0.001439971000,
    0.000999949300, 0.000690078600, 0.000476021300, 0.000332301100, 0.000234826100,
    0.000166150500, 0.000117413000, 0.000083075270, 0.000058706520, 0.000041509940,
    0.000029353260, 0.000020673830, 0.000014559770, 0.000010253980, 0.000007221456,
    0.000005085868, 0.000003581652, 0.000002522525, 0.000001776509, 0.000001251141};

const double cie_y[CIE_SAMPLES] = {
    0.000003917000, 0.000006965000, 0.000012390000, 0.000022020000, 0.000039000000,
    0.000064000000, 0.000120000000, 0.000217000000, 0.000396000000, 0.000640000000,
    0.001210000000, 0.002180000000, 0.004000000000, 0.007300000000, 0.011600000000,
    0.016840000000, 0.023000000000, 0.029800000000, 0.038000000000, 0.048000000000,
    0.060000000000, 0.073900000000, 0.090980000000, 0.112600000000, 0.139020000000,
    0.169300000000, 0.208020000000, 0.258600000000, 0.323000000000, 0.407300000000,
    0.503000000000, 0.608200000000, 0.710000000000, 0.793200000000, 0.862000000000,
    0.914850100000, 0.954000000000, 0.980300000000, 0.994950100000, 1.000000000000,
    0.995000000000, 0.978600000000, 0.952000000000, 0.915400000000, 0.870000000000,
    0.816300000000, 0.757000000000, 0.694900000000, 0.631000000000, 0.566800000000,
    0.503000000000, 0.441200000000, 0.381000000000, 0.321000000000, 0.265000000000,
    0.217000000000, 0.175000000000, 0.138200000000, 0.107000000000, 0.081600000000,
    0.061000000000, 0.044580000000, 0.032000000000, 0.023200000000, 0.017000000000,
    0.011920000000, 0.008210000000, 0.005723000000, 0.004102000000, 0.002929000000,
    0.002091000000, 0.001484000000, 0.001047000000, 0.000740000000, 0.000520000000,
    0.000361100000, 0.000249200000, 0.000171900000, 0.000120000000, 0.000084800000,
    0.000060000000, 0.000042400000, 0.000030000000, 0.000021200000, 0.000014990000,
    0.000010600000, 0.000007465700, 0.000005257800, 0.000003702900, 0.000002607800,
    0.000001836600, 0.000001293400, 0.000000910930, 0.000000641530, 0.000000451810};

const double cie_z[CIE_SAMPLES] = {
    0.000606100000, 0.001086000000, 0.001946000000, 0.003486000000, 0.006450001000,
    0.010549990000, 0.020050010000, 0.036210000000, 0.067850010000, 0.110200000000,
    0.207400000000, 0.371300000000, 0.645600000000, 1.039050100000, 1.385600000000,
    1.622960000000, 1.747060000000, 1.782600000000, 1.772110000000, 1.744100000000,
    1.669200000000, 1.528100000000, 1.287640000000, 1.041900000000, 0.812950100000,
    0.616200000000, 0.465180000000, 0.353300000000, 0.272000000000, 0.212300000000,
    0.158200000000, 0.111700000000, 0.078249990000, 0.057250010000, 0.042160000000,
    0.029840000000, 0.020300000000, 0.013400000000, 0.008749999000, 0.005749999000,
    0.003900000000, 0.002749999000, 0.002100000000, 0.001800000000, 0.001650001000,
    0.001400000000, 0.001100000000, 0.001000000000, 0.000800000000, 0.000600000000,
    0.000340000000, 0.000240000000, 0.000190000000, 0.000100000000, 0.000049999990,
    0.000030000000, 0.000020000000, 0.000010000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000};

#define N(x) (x / 10566.864005283874576)

const double cie_d65[CIE_SAMPLES] = {
    N(46.6383), N(49.3637), N(52.0891), N(51.0323), N(49.9755), N(52.3118), N(54.6482),
    N(68.7015), N(82.7549), N(87.1204), N(91.486),  N(92.4589), N(93.4318), N(90.057),
    N(86.6823), N(95.7736), N(104.865), N(110.936), N(117.008), N(117.41),  N(117.812),
    N(116.336), N(114.861), N(115.392), N(115.923), N(112.367), N(108.811), N(109.082),
    N(109.354), N(108.578), N(107.802), N(106.296), N(104.79),  N(106.239), N(107.689),
    N(106.047), N(104.405), N(104.225), N(104.046), N(102.023), N(100.0),   N(98.1671),
    N(96.3342), N(96.0611), N(95.788),  N(92.2368), N(88.6856), N(89.3459), N(90.0062),
    N(89.8026), N(89.5991), N(88.6489), N(87.6987), N(85.4936), N(83.2886), N(83.4939),
    N(83.6992), N(81.863),  N(80.0268), N(80.1207), N(80.2146), N(81.2462), N(82.2778),
    N(80.281),  N(78.2842), N(74.0027), N(69.7213), N(70.6652), N(71.6091), N(72.979),
    N(74.349),  N(67.9765), N(61.604),  N(65.7448), N(69.8856), N(72.4863), N(75.087),
    N(69.3398), N(63.5927), N(55.0054), N(46.4182), N(56.6118), N(66.8054), N(65.0941),
    N(63.3828), N(63.8434), N(64.304),  N(61.8779), N(59.4519), N(55.7054), N(51.959),
    N(54.6998), N(57.4406), N(58.8765), N(60.3125)};

#undef N

#define N(x) (x / 106.8)
const double cie_e[CIE_SAMPLES] = {
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0)};
#undef N

#define N(x) (x / 10503.2)

const double cie_d50[CIE_SAMPLES] = {
    N(23.942000),  N(25.451000),  N(26.961000),  N(25.724000),  N(24.488000),
    N(27.179000),  N(29.871000),  N(39.589000),  N(49.308000),  N(52.910000),
    N(56.513000),  N(58.273000),  N(60.034000),  N(58.926000),  N(57.818000),
    N(66.321000),  N(74.825000),  N(81.036000),  N(87.247000),  N(88.930000),
    N(90.612000),  N(90.990000),  N(91.368000),  N(93.238000),  N(95.109000),
    N(93.536000),  N(91.963000),  N(93.843000),  N(95.724000),  N(96.169000),
    N(96.613000),  N(96.871000),  N(97.129000),  N(99.614000),  N(102.099000),
    N(101.427000), N(100.755000), N(101.536000), N(102.317000), N(101.159000),
    N(100.000000), N(98.868000),  N(97.735000),  N(98.327000),  N(98.918000),
    N(96.208000),  N(93.499000),  N(95.593000),  N(97.688000),  N(98.478000),
    N(99.269000),  N(99.155000),  N(99.042000),  N(97.382000),  N(95.722000),
    N(97.290000),  N(98.857000),  N(97.262000),  N(95.667000),  N(96.929000),
    N(98.190000),  N(100.597000), N(103.003000), N(101.068000), N(99.133000),
    N(93.257000),  N(87.381000),  N(89.492000),  N(91.604000),  N(92.246000),
    N(92.889000),  N(84.872000),  N(76.854000),  N(81.683000),  N(86.511000),
    N(89.546000),  N(92.580000),  N(85.405000),  N(78.230000),  N(67.961000),
    N(57.692000),  N(70.307000),  N(82.923000),  N(80.599000),  N(78.274000),
    N(0),          N(0),          N(0),          N(0),          N(0),
    N(0),          N(0),          N(0),          N(0)};

#undef N

#define N(x) (x / 10536.3)

const double cie_d60[CIE_SAMPLES] = {
    N(38.683115),  N(41.014457),  N(42.717548),  N(42.264182),  N(41.454941),
    N(41.763698),  N(46.605319),  N(59.226938),  N(72.278594),  N(78.231500),
    N(80.440600),  N(82.739580),  N(82.915027),  N(79.009168),  N(77.676264),
    N(85.163609),  N(95.681274),  N(103.267764), N(107.954821), N(109.777964),
    N(109.559187), N(108.418402), N(107.758141), N(109.071548), N(109.671404),
    N(106.734741), N(103.707873), N(103.981942), N(105.232199), N(105.235867),
    N(104.427667), N(103.052881), N(102.522934), N(104.371416), N(106.052671),
    N(104.948900), N(103.315154), N(103.416286), N(103.538599), N(102.099304),
    N(100.000000), N(97.992725),  N(96.751421),  N(97.102402),  N(96.712823),
    N(93.174457),  N(89.921479),  N(90.351933),  N(91.999793),  N(92.384009),
    N(92.098710),  N(91.722859),  N(90.646003),  N(88.327552),  N(86.526483),
    N(87.034239),  N(87.579186),  N(85.884584),  N(83.976140),  N(83.743140),
    N(84.724074),  N(86.450818),  N(87.493491),  N(86.546330),  N(83.483070),
    N(78.268785),  N(74.172451),  N(74.275184),  N(76.620385),  N(79.423856),
    N(79.051849),  N(71.763360),  N(65.471371),  N(67.984085),  N(74.106079),
    N(78.556612),  N(79.527120),  N(75.584935),  N(67.307163),  N(55.275106),
    N(49.273538),  N(59.008629),  N(70.892412),  N(70.950115),  N(67.163996),
    N(67.445480),  N(68.171371),  N(66.466636),  N(62.989809),  N(58.067786),
    N(54.990892),  N(56.915942),  N(60.825601),  N(62.987850)};

#undef N

const double xyz_to_srgb[3][3] = {{3.240479, -1.537150, -0.498535},
                                  {-0.969256, 1.875991, 0.041556},
                                  {0.055648, -0.204043, 1.057311}};

const double srgb_to_xyz[3][3] = {{0.412453, 0.357580, 0.180423},
                                  {0.212671, 0.715160, 0.072169},
                                  {0.019334, 0.119193, 0.950227}};

const double xyz_to_xyz[3][3] = {
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
};

const double xyz_to_ergb[3][3] = {
    {2.689989, -1.276020, -0.413844},
    {-1.022095, 1.978261, 0.043821},
    {0.061203, -0.224411, 1.162859},
};

const double ergb_to_xyz[3][3] = {
    {0.496859, 0.339094, 0.164047},
    {0.256193, 0.678188, 0.065619},
    {0.023290, 0.113031, 0.863978},
};

const double xyz_to_prophoto_rgb[3][3] = {{1.3459433, -0.2556075, -0.0511118},
                                          {-0.5445989, 1.5081673, 0.0205351},
                                          {0.0000000, 0.0000000, 1.2118128}};

const double prophoto_rgb_to_xyz[3][3] = {{0.7976749, 0.1351917, 0.0313534},
                                          {0.2880402, 0.7118741, 0.0000857},
                                          {0.0000000, 0.0000000, 0.8252100}};

const double xyz_to_aces2065_1[3][3] = {{1.0498110175, 0.0000000000, -0.0000974845},
                                        {-0.4959030231, 1.3733130458, 0.0982400361},
                                        {0.0000000000, 0.0000000000, 0.9912520182}};

const double aces2065_1_to_xyz[3][3] = {{0.9525523959, 0.0000000000, 0.0000936786},
                                        {0.3439664498, 0.7281660966, -0.0721325464},
                                        {0.0000000000, 0.0000000000, 1.0088251844}};

const double xyz_to_rec2020[3][3] = {{1.7166511880, -0.3556707838, -0.2533662814},
                                     {-0.6666843518, 1.6164812366, 0.0157685458},
                                     {0.0176398574, -0.0427706133, 0.9421031212}};

const double rec2020_to_xyz[3][3] = {{0.6369580483, 0.1446169036, 0.1688809752},
                                     {0.2627002120, 0.6779980715, 0.0593017165},
                                     {0.0000000000, 0.0280726930, 1.0609850577}};

const double xyz_to_dcip3[3][3] = {{2.4931748, -0.93126315, -0.40265882},
                                   {-0.82950425, 1.7626965, 0.023625137},
                                   {0.035853732, -0.07618918, 0.9570952}};
const double dcip3_to_xyz[3][3] = {{0.48663378, 0.26566276, 0.19817366},
                                   {0.22900413, 0.69172573, 0.079269454},
                                   {0., 0.04511256, 1.0437145}};

double cie_interp(const double *data, double x) {
    x -= CIE_LAMBDA_MIN;
    x *= (CIE_SAMPLES - 1) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);
    int offset = (int)x;
    if (offset < 0)
        offset = 0;
    if (offset > CIE_SAMPLES - 2)
        offset = CIE_SAMPLES - 2;
    double weight = x - offset;
    return (1.0 - weight) * data[offset] + weight * data[offset + 1];
}

// LU decomposition & triangular solving code lifted from Wikipedia

/* INPUT: A - array of pointers to rows of a square matrix having dimension N
 *        Tol - small tolerance number to detect failure when the matrix is near
 * degenerate OUTPUT: Matrix A is changed, it contains both matrices L-E and U
 * as A=(L-E)+U such that P*A=L*U. The permutation matrix is not stored as a
 * matrix, but in an integer vector P of size N+1 containing column indexes
 * where the permutation matrix has "1". The last element P[N]=S+N, where S is
 * the number of row exchanges needed for determinant computation, det(P)=(-1)^S
 */
int LUPDecompose(double **A, int N, double Tol, int *P) {
    int i, j, k, imax;
    double maxA, *ptr, absA;

    for (i = 0; i <= N; i++)
        P[i] = i;  // Unit permutation matrix, P[N] initialized with N

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        for (k = i; k < N; k++)
            if ((absA = fabs(A[k][i])) > maxA) {
                maxA = absA;
                imax = k;
            }

        if (maxA < Tol)
            return 0;  // failure, matrix is degenerate

        if (imax != i) {
            // pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // pivoting rows of A
            ptr = A[i];
            A[i] = A[imax];
            A[imax] = ptr;

            // counting pivots starting from N (for determinant)
            P[N]++;
        }

        for (j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];

            for (k = i + 1; k < N; k++)
                A[j][k] -= A[j][i] * A[i][k];
        }
    }

    return 1;  // decomposition done
}

/* INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
 * OUTPUT: x - solution vector of A*x=b
 */
void LUPSolve(double **const A, const int *P, const double *b, int N, double *x) {
    for (int i = 0; i < N; i++) {
        x[i] = b[P[i]];

        for (int k = 0; k < i; k++)
            x[i] -= A[i][k] * x[k];
    }

    for (int i = N - 1; i >= 0; i--) {
        for (int k = i + 1; k < N; k++)
            x[i] -= A[i][k] * x[k];

        x[i] = x[i] / A[i][i];
    }
}

#if defined(_OPENMP)
#define RGB2SPEC_USE_OPENMP 1
#elif defined(__APPLE__)
#define RGB2SPEC_USE_GCD 1
#include <dispatch/dispatch.h>
#endif

/// Discretization of quadrature scheme
#define CIE_FINE_SAMPLES ((CIE_SAMPLES - 1) * 3 + 1)
#define RGB2SPEC_EPSILON 1e-4

/// Precomputed tables for fast spectral -> RGB conversion
double lambda_tbl[CIE_FINE_SAMPLES], rgb_tbl[3][CIE_FINE_SAMPLES], rgb_to_xyz[3][3],
    xyz_to_rgb[3][3], xyz_whitepoint[3];

/// Currently supported gamuts
enum Gamut {
    SRGB,
    ProPhotoRGB,
    ACES2065_1,
    REC2020,
    ERGB,
    XYZ,
    DCI_P3,
    NO_GAMUT,
};

double sigmoid(double x) {
    return 0.5 * x / std::sqrt(1.0 + x * x) + 0.5;
}

double smoothstep(double x) {
    return x * x * (3.0 - 2.0 * x);
}

double sqr(double x) {
    return x * x;
}

void cie_lab(double *p) {
    double X = 0.0, Y = 0.0, Z = 0.0, Xw = xyz_whitepoint[0], Yw = xyz_whitepoint[1],
           Zw = xyz_whitepoint[2];

    for (int j = 0; j < 3; ++j) {
        X += p[j] * rgb_to_xyz[0][j];
        Y += p[j] * rgb_to_xyz[1][j];
        Z += p[j] * rgb_to_xyz[2][j];
    }

    auto f = [](double t) -> double {
        double delta = 6.0 / 29.0;
        if (t > delta * delta * delta)
            return cbrt(t);
        else
            return t / (delta * delta * 3.0) + (4.0 / 29.0);
    };

    p[0] = 116.0 * f(Y / Yw) - 16.0;
    p[1] = 500.0 * (f(X / Xw) - f(Y / Yw));
    p[2] = 200.0 * (f(Y / Yw) - f(Z / Zw));
}

/**
 * This function precomputes tables used to convert arbitrary spectra
 * to RGB (either sRGB or ProPhoto RGB)
 *
 * A composite quadrature rule integrates the CIE curves, reflectance, and
 * illuminant spectrum over each 5nm segment in the 360..830nm range using
 * Simpson's 3/8 rule (4th-order accurate), which evaluates the integrand at
 * four positions per segment. While the CIE curves and illuminant spectrum are
 * linear over the segment, the reflectance could have arbitrary behavior,
 * hence the extra precations.
 */
void init_tables(Gamut gamut) {
    memset(rgb_tbl, 0, sizeof(rgb_tbl));
    memset(xyz_whitepoint, 0, sizeof(xyz_whitepoint));

    double h = (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN) / (CIE_FINE_SAMPLES - 1);

    const double *illuminant = nullptr;

    switch (gamut) {
    case SRGB:
        illuminant = cie_d65;
        memcpy(xyz_to_rgb, xyz_to_srgb, sizeof(double) * 9);
        memcpy(rgb_to_xyz, srgb_to_xyz, sizeof(double) * 9);
        break;

    case ERGB:
        illuminant = cie_e;
        memcpy(xyz_to_rgb, xyz_to_ergb, sizeof(double) * 9);
        memcpy(rgb_to_xyz, ergb_to_xyz, sizeof(double) * 9);
        break;

    case XYZ:
        illuminant = cie_e;
        memcpy(xyz_to_rgb, xyz_to_xyz, sizeof(double) * 9);
        memcpy(rgb_to_xyz, xyz_to_xyz, sizeof(double) * 9);
        break;

    case ProPhotoRGB:
        illuminant = cie_d50;
        memcpy(xyz_to_rgb, xyz_to_prophoto_rgb, sizeof(double) * 9);
        memcpy(rgb_to_xyz, prophoto_rgb_to_xyz, sizeof(double) * 9);
        break;

    case ACES2065_1:
        illuminant = cie_d60;
        memcpy(xyz_to_rgb, xyz_to_aces2065_1, sizeof(double) * 9);
        memcpy(rgb_to_xyz, aces2065_1_to_xyz, sizeof(double) * 9);
        break;

    case REC2020:
        illuminant = cie_d65;
        memcpy(xyz_to_rgb, xyz_to_rec2020, sizeof(double) * 9);
        memcpy(rgb_to_xyz, rec2020_to_xyz, sizeof(double) * 9);
        break;

    case DCI_P3:
        illuminant = cie_d65;
        memcpy(xyz_to_rgb, xyz_to_dcip3, sizeof(double) * 9);
        memcpy(rgb_to_xyz, dcip3_to_xyz, sizeof(double) * 9);
        break;

    default:
        throw std::runtime_error("init_gamut(): invalid/unsupported gamut.");
    }

    for (int i = 0; i < CIE_FINE_SAMPLES; ++i) {
        double lambda = CIE_LAMBDA_MIN + i * h;

        double xyz[3] = {cie_interp(cie_x, lambda), cie_interp(cie_y, lambda),
                         cie_interp(cie_z, lambda)},
               I = cie_interp(illuminant, lambda);

        double weight = 3.0 / 8.0 * h;
        if (i == 0 || i == CIE_FINE_SAMPLES - 1)
            ;
        else if ((i - 1) % 3 == 2)
            weight *= 2.f;
        else
            weight *= 3.f;

        lambda_tbl[i] = lambda;
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
                rgb_tbl[k][i] += xyz_to_rgb[k][j] * xyz[j] * I * weight;

        for (int i = 0; i < 3; ++i)
            xyz_whitepoint[i] += xyz[i] * I * weight;
    }
}

void eval_residual(const double *coeffs, const double *rgb, double *residual) {
    double out[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < CIE_FINE_SAMPLES; ++i) {
        /* Scale lambda to 0..1 range */
        double lambda =
            (lambda_tbl[i] - CIE_LAMBDA_MIN) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);

        /* Polynomial */
        double x = 0.0;
        for (int i = 0; i < 3; ++i)
            x = x * lambda + coeffs[i];

        /* Sigmoid */
        double s = sigmoid(x);

        /* Integrate against precomputed curves */
        for (int j = 0; j < 3; ++j)
            out[j] += rgb_tbl[j][i] * s;
    }
    cie_lab(out);
    memcpy(residual, rgb, sizeof(double) * 3);
    cie_lab(residual);

    for (int j = 0; j < 3; ++j)
        residual[j] -= out[j];
}

void eval_jacobian(const double *coeffs, const double *rgb, double **jac) {
    double r0[3], r1[3], tmp[3];

    for (int i = 0; i < 3; ++i) {
        memcpy(tmp, coeffs, sizeof(double) * 3);
        tmp[i] -= RGB2SPEC_EPSILON;
        eval_residual(tmp, rgb, r0);

        memcpy(tmp, coeffs, sizeof(double) * 3);
        tmp[i] += RGB2SPEC_EPSILON;
        eval_residual(tmp, rgb, r1);

        for (int j = 0; j < 3; ++j)
            jac[j][i] = (r1[j] - r0[j]) * 1.0 / (2 * RGB2SPEC_EPSILON);
    }
}

void gauss_newton(const double rgb[3], double coeffs[3], int it = 15) {
    double r = 0;
    for (int i = 0; i < it; ++i) {
        double J0[3], J1[3], J2[3], *J[3] = {J0, J1, J2};

        double residual[3];

        eval_residual(coeffs, rgb, residual);
        eval_jacobian(coeffs, rgb, J);

        int P[4];
        int rv = LUPDecompose(J, 3, 1e-15, P);
        if (rv != 1) {
            std::cout << "RGB " << rgb[0] << " " << rgb[1] << " " << rgb[2] << std::endl;
            std::cout << "-> " << coeffs[0] << " " << coeffs[1] << " " << coeffs[2]
                      << std::endl;
            throw std::runtime_error("LU decomposition failed!");
        }

        double x[3];
        LUPSolve(J, P, residual, 3, x);

        r = 0.0;
        for (int j = 0; j < 3; ++j) {
            coeffs[j] -= x[j];
            r += residual[j] * residual[j];
        }
        double max = std::max(std::max(coeffs[0], coeffs[1]), coeffs[2]);

        if (max > 200) {
            for (int j = 0; j < 3; ++j)
                coeffs[j] *= 200 / max;
        }

        if (r < 1e-6)
            break;
    }
}

static Gamut parse_gamut(const char *str) {
    if (!strcasecmp(str, "sRGB"))
        return SRGB;
    if (!strcasecmp(str, "eRGB"))
        return ERGB;
    if (!strcasecmp(str, "XYZ"))
        return XYZ;
    if (!strcasecmp(str, "ProPhotoRGB"))
        return ProPhotoRGB;
    if (!strcasecmp(str, "ACES2065_1"))
        return ACES2065_1;
    if (!strcasecmp(str, "REC2020"))
        return REC2020;
    if (!strcasecmp(str, "DCI_P3"))
        return DCI_P3;
    return NO_GAMUT;
}

/* hack: below is a copy of enough of util/parallel.* to be able to run
   ParallelFor to generate the tables. Note that we don't want to #include
   <util/parallel.h>, since we'd end up spending lots of time regenerating
   these tables whenever that header file changed.
 */

void ParallelFor(int64_t start, int64_t end, std::function<void(int64_t, int64_t)> func,
                 const char *progressName = nullptr);

inline void ParallelFor(int64_t start, int64_t end, std::function<void(int64_t)> func,
                        const char *progressName = nullptr) {
    ParallelFor(
        start, end,
        [&func](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i)
                func(i);
        },
        progressName);
}

class ParallelJob {
  public:
    virtual ~ParallelJob() { assert(removed); }

    // *lock should be locked going in and and unlocked coming out.
    virtual void RunStep(std::unique_lock<std::mutex> *lock) = 0;
    virtual bool HaveWork() const = 0;

    bool Finished() const { return !HaveWork() && activeWorkers == 0; }

  private:
    friend class ThreadPool;

    ParallelJob *prev = nullptr, *next = nullptr;
    int activeWorkers = 0;
    bool removed = false;
};

class ThreadPool {
  public:
    explicit ThreadPool(int nThreads);
    ~ThreadPool();

    size_t size() const { return threads.size(); }

    std::unique_lock<std::mutex> AddToJobList(ParallelJob *job);
    void RemoveFromJobList(ParallelJob *job);

    void WorkOrWait(std::unique_lock<std::mutex> *lock);

  private:
    void workerFunc(int tIndex);

    ParallelJob *jobList = nullptr;
    // Protects jobList
    mutable std::mutex jobListMutex;
    // Signaled both when a new job is added to the list and when a job has
    // finished.
    std::condition_variable jobListCondition;

    std::vector<std::thread> threads;
    bool shutdownThreads = false;
};

static std::unique_ptr<ThreadPool> threadPool;

int AvailableCores() {
    return std::max<int>(1, std::thread::hardware_concurrency());
}

int RunningThreads() {
    return threadPool ? (1 + threadPool->size()) : 1;
}

ThreadPool::ThreadPool(int nThreads) {
    // Launch one fewer worker thread than the total number we want doing
    // work, since the main thread helps out, too.
    for (int i = 0; i < nThreads - 1; ++i)
        threads.push_back(std::thread(&ThreadPool::workerFunc, this, i + 1));
}

ThreadPool::~ThreadPool() {
    if (threads.empty())
        return;

    {
        std::lock_guard<std::mutex> lock(jobListMutex);
        shutdownThreads = true;
        jobListCondition.notify_all();
    }

    for (std::thread &thread : threads)
        thread.join();
}

std::unique_lock<std::mutex> ThreadPool::AddToJobList(ParallelJob *job) {
    std::unique_lock<std::mutex> lock(jobListMutex);
    if (jobList != nullptr)
        jobList->prev = job;
    job->next = jobList;
    jobList = job;
    jobListCondition.notify_all();
    return lock;
}

void ThreadPool::RemoveFromJobList(ParallelJob *job) {
    assert(!job->removed);

    if (job->prev != nullptr) {
        job->prev->next = job->next;
    } else {
        assert(jobList == job);
        jobList = job->next;
    }
    if (job->next != nullptr)
        job->next->prev = job->prev;

    job->removed = true;
}

void ThreadPool::workerFunc(int tIndex) {
    std::unique_lock<std::mutex> lock(jobListMutex);
    while (!shutdownThreads)
        WorkOrWait(&lock);
}

void ThreadPool::WorkOrWait(std::unique_lock<std::mutex> *lock) {
    assert(lock->owns_lock());

    ParallelJob *job = jobList;
    while ((job != nullptr) && !job->HaveWork())
        job = job->next;
    if (job != nullptr) {
        // Run a chunk of loop iterations for _loop_
        job->activeWorkers++;

        job->RunStep(lock);

        assert(!lock->owns_lock());
        lock->lock();

        // Update _loop_ to reflect completion of iterations
        job->activeWorkers--;

        if (job->Finished())
            jobListCondition.notify_all();
    } else
        // Wait for something to change (new work, or this loop being
        // finished).
        jobListCondition.wait(*lock);
}

class ParallelForLoop1D : public ParallelJob {
  public:
    ParallelForLoop1D(int64_t start, int64_t end, int chunkSize,
                      std::function<void(int64_t, int64_t)> func)
        : func(std::move(func)), nextIndex(start), maxIndex(end), chunkSize(chunkSize) {}

    bool HaveWork() const { return nextIndex < maxIndex; }
    void RunStep(std::unique_lock<std::mutex> *lock);

  private:
    std::function<void(int64_t, int64_t)> func;
    int64_t nextIndex;
    int64_t maxIndex;
    int chunkSize;
};

void ParallelForLoop1D::RunStep(std::unique_lock<std::mutex> *lock) {
    // Find the set of loop iterations to run next
    int64_t indexStart = nextIndex;
    int64_t indexEnd = std::min(indexStart + chunkSize, maxIndex);

    // Update _loop_ to reflect iterations this thread will run
    nextIndex = indexEnd;

    if (!HaveWork())
        threadPool->RemoveFromJobList(this);

    lock->unlock();

    // Run loop indices in _[indexStart, indexEnd)_
    func(indexStart, indexEnd);
}

void ParallelFor(int64_t start, int64_t end, std::function<void(int64_t, int64_t)> func,
                 const char *progressName) {
    assert(threadPool);

    int64_t chunkSize = std::max<int64_t>(1, (end - start) / (8 * RunningThreads()));

    // Create and enqueue _ParallelJob_ for this loop
    ParallelForLoop1D loop(start, end, chunkSize, std::move(func));
    std::unique_lock<std::mutex> lock = threadPool->AddToJobList(&loop);

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished())
        threadPool->WorkOrWait(&lock);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Syntax: rgb2spec_opt <resolution> <output> [<gamut>]\n"
               "where <gamut> is one of "
               "sRGB,eRGB,XYZ,ProPhotoRGB,ACES2065_1,REC2020\n");
        exit(-1);
    }
    Gamut gamut = SRGB;
    if (argc > 3)
        gamut = parse_gamut(argv[3]);
    if (gamut == NO_GAMUT) {
        fprintf(stderr, "Could not parse gamut `%s'!\n", argv[3]);
        exit(-1);
    }
    init_tables(gamut);

    const int res = atoi(argv[1]);
    if (res == 0) {
        printf("Invalid resolution!\n");
        exit(-1);
    }

    int nThreads = AvailableCores();
    threadPool = std::make_unique<ThreadPool>(nThreads);

    printf("Optimizing %s spectra...\n", argv[3]);
    fflush(stdout);

    float *scale = new float[res];
    for (int k = 0; k < res; ++k)
        scale[k] = (float)smoothstep(smoothstep(k / double(res - 1)));

    size_t bufsize = 3 * 3 * res * res * res;
    float *out = new float[bufsize];

    for (int l = 0; l < 3; ++l) {
        ParallelFor(0, res, [&](size_t j) {
            const double y = j / double(res - 1);
            fflush(stdout);
            for (int i = 0; i < res; ++i) {
                const double x = i / double(res - 1);
                double coeffs[3], rgb[3];
                memset(coeffs, 0, sizeof(double) * 3);

                int start = res / 5;

                for (int k = start; k < res; ++k) {
                    double b = (double)scale[k];

                    rgb[l] = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;

                    gauss_newton(rgb, coeffs);

                    double c0 = 360.0, c1 = 1.0 / (830.0 - 360.0);
                    double A = coeffs[0], B = coeffs[1], C = coeffs[2];

                    int idx = ((l * res + k) * res + j) * res + i;

                    out[3 * idx + 0] = float(A * (sqr(c1)));
                    out[3 * idx + 1] = float(B * c1 - 2 * A * c0 * (sqr(c1)));
                    out[3 * idx + 2] = float(C - B * c0 * c1 + A * (sqr(c0 * c1)));
                    // out[3*idx + 2] = resid;
                }

                memset(coeffs, 0, sizeof(double) * 3);
                for (int k = start; k >= 0; --k) {
                    double b = (double)scale[k];

                    rgb[l] = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;

                    gauss_newton(rgb, coeffs);

                    double c0 = 360.0, c1 = 1.0 / (830.0 - 360.0);
                    double A = coeffs[0], B = coeffs[1], C = coeffs[2];

                    int idx = ((l * res + k) * res + j) * res + i;

                    out[3 * idx + 0] = float(A * (sqr(c1)));
                    out[3 * idx + 1] = float(B * c1 - 2 * A * c0 * (sqr(c1)));
                    out[3 * idx + 2] = float(C - B * c0 * c1 + A * (sqr(c0 * c1)));
                    // out[3*idx + 2] = resid;
                }
            }
        });
    }

    FILE *f = fopen(argv[2], "w");
    if (f == nullptr)
        throw std::runtime_error("Could not create file!");
    fprintf(f, "namespace pbrt {\n");
    fprintf(f, "extern const int %sToSpectrumTable_Res = %d;\n", argv[3], res);
    fprintf(f, "extern const float %sToSpectrumTable_Scale[%d] = {\n", argv[3], res);
    for (int i = 0; i < res; ++i)
        fprintf(f, "%.9g, ", scale[i]);
    fprintf(f, "};\n");
    fprintf(f, "extern const float %sToSpectrumTable_Data[3][%d][%d][%d][3] = {\n",
            argv[3], res, res, res);
    const float *ptr = out;
    for (int maxc = 0; maxc < 3; ++maxc) {
        fprintf(f, "{ ");
        for (int z = 0; z < res; ++z) {
            fprintf(f, "{ ");
            for (int y = 0; y < res; ++y) {
                fprintf(f, "{ ");
                for (int x = 0; x < res; ++x) {
                    fprintf(f, "{ ");
                    for (int c = 0; c < 3; ++c)
                        fprintf(f, "%.9g, ", *ptr++);
                    fprintf(f, "}, ");
                }
                fprintf(f, "},\n    ");
            }
            fprintf(f, "}, ");
        }
        fprintf(f, "}, ");
    }
    fprintf(f, "};\n");
    fprintf(f, "} // namespace pbrt\n");
    fclose(f);

    threadPool.reset();
}
