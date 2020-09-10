// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/cameras.h>

#include <pbrt/base/medium.h>
#include <pbrt/bsdf.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>

#include <algorithm>

namespace pbrt {

// CameraTransform Method Definitions
CameraTransform::CameraTransform(const AnimatedTransform &worldFromCamera) {
    switch (Options->renderingSpace) {
    case RenderingCoordinateSystem::Camera: {
        // Compute _worldFromRender_ for camera-space rendering
        Float tMid = (worldFromCamera.startTime + worldFromCamera.endTime) / 2;
        worldFromRender = worldFromCamera.Interpolate(tMid);
        break;
    }
    case RenderingCoordinateSystem::CameraWorld: {
        // Compute _worldFromRender_ for camera-world space rendering
        Float tMid = (worldFromCamera.startTime + worldFromCamera.endTime) / 2;
        Point3f pCamera = worldFromCamera(Point3f(0, 0, 0), tMid);
        worldFromRender = Translate(Vector3f(pCamera));
        break;
    }
    case RenderingCoordinateSystem::World: {
        // Compute _worldFromRender_ for world-space rendering
        worldFromRender = Transform();
        break;
    }
    default:
        LOG_FATAL("Unhandled rendering coordinate space");
    }
    // Compute _renderFromCamera_ transformation
    Transform renderFromWorld = Inverse(worldFromRender);
    Transform rfc[2] = {renderFromWorld * worldFromCamera.startTransform,
                        renderFromWorld * worldFromCamera.endTransform};
    renderFromCamera = AnimatedTransform(rfc[0], worldFromCamera.startTime, rfc[1],
                                         worldFromCamera.endTime);
}

std::string CameraTransform::ToString() const {
    return StringPrintf("[ CameraTransform renderFromCamera: %s worldFromRender: %s ]",
                        renderFromCamera, worldFromRender);
}

// Camera Method Definitions
pstd::optional<CameraRayDifferential> CameraHandle::GenerateRayDifferential(
    const CameraSample &sample, SampledWavelengths &lambda) const {
    auto gen = [&](auto ptr) { return ptr->GenerateRayDifferential(sample, lambda); };
    return Dispatch(gen);
}

void CameraHandle::ApproximatedPdxy(const SurfaceInteraction &si) const {
    auto approx = [&](auto ptr) { return ptr->ApproximatedPdxy(si); };
    return Dispatch(approx);
}

SampledSpectrum CameraHandle::We(const Ray &ray, SampledWavelengths &lambda,
                                 Point2f *pRaster2) const {
    auto we = [&](auto ptr) { return ptr->We(ray, lambda, pRaster2); };
    return Dispatch(we);
}

void CameraHandle::PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    auto pdf = [&](auto ptr) { return ptr->PDF_We(ray, pdfPos, pdfDir); };
    return Dispatch(pdf);
}

pstd::optional<CameraWiSample> CameraHandle::SampleWi(const Interaction &ref,
                                                      const Point2f &u,
                                                      SampledWavelengths &lambda) const {
    auto sample = [&](auto ptr) { return ptr->SampleWi(ref, u, lambda); };
    return Dispatch(sample);
}

void CameraHandle::InitMetadata(ImageMetadata *metadata) const {
    auto init = [&](auto ptr) { return ptr->InitMetadata(metadata); };
    return DispatchCPU(init);
}

std::string CameraHandle::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

// CameraBase Method Definitions
CameraBase::CameraBase(CameraBaseParameters p)
    : cameraTransform(p.cameraTransform),
      shutterOpen(p.shutterOpen),
      shutterClose(p.shutterClose),
      film(p.film),
      medium(p.medium) {
    if (cameraTransform.CameraFromRenderHasScale())
        Warning("Scaling detected in world-to-camera transformation!\n"
                "The system has numerous assumptions, implicit and explicit,\n"
                "that this transform will have no scale factors in it.\n"
                "Proceed at your own risk; your image may have errors or\n"
                "the system may crash as a result of this.");
}

pstd::optional<CameraRayDifferential> CameraBase::GenerateRayDifferential(
    CameraHandle camera, const CameraSample &sample, SampledWavelengths &lambda) {
    // Generate regular camera ray _cr_ for ray differential
    pstd::optional<CameraRay> cr = camera.GenerateRay(sample, lambda);
    if (!cr)
        return {};
    RayDifferential rd(cr->ray);

    // Find camera ray after shifting one pixel in the $x$ direction
    pstd::optional<CameraRay> rx;
    for (Float eps : {.05f, -.05f}) {
        CameraSample sshift = sample;
        sshift.pFilm.x += eps;
        // Try to generate ray with _sshift_ and compute $x$ differential
        if (rx = camera.GenerateRay(sshift, lambda); rx) {
            rd.rxOrigin = rd.o + (rx->ray.o - rd.o) / eps;
            rd.rxDirection = rd.d + (rx->ray.d - rd.d) / eps;
            break;
        }
    }

    // Find camera ray after shifting one pixel in the $y$ direction
    pstd::optional<CameraRay> ry;
    for (Float eps : {.05f, -.05f}) {
        CameraSample sshift = sample;
        sshift.pFilm.y += eps;
        if (ry = camera.GenerateRay(sshift, lambda); ry) {
            rd.ryOrigin = rd.o + (ry->ray.o - rd.o) / eps;
            rd.ryDirection = rd.d + (ry->ray.d - rd.d) / eps;
            break;
        }
    }

    // Return approximate ray differential and weight
    rd.hasDifferentials = rx && ry;
    return CameraRayDifferential{rd, cr->weight};
}

void CameraBase::ApproximatedPdxy(const SurfaceInteraction &si) const {
    Point3f pc = CameraFromRender(si.p(), si.time);
    Float dist = Distance(pc, Point3f(0, 0, 0));

    Frame f = Frame::FromZ(si.n);
    // ray plane:
    // (0,0,0) + minPosDifferential + ((0,0,1) + minDirDifferantial)) * t = (x,
    // x, dist)
    Float tx = (dist - minPosDifferentialX.z) / (1 + minDirDifferentialX.z);
    // 0.5 factor to sharpen them up slightly (could be / should be based
    // on spp?)
    si.dpdx = .5f * f.FromLocal(minPosDifferentialX + tx * minDirDifferentialX);
    Float ty = (dist - minPosDifferentialY.z) / (1 + minDirDifferentialY.z);
    si.dpdy = .5f * f.FromLocal(minPosDifferentialY + ty * minDirDifferentialY);
}

void CameraBase::InitMetadata(ImageMetadata *metadata) const {
    metadata->cameraFromWorld = cameraTransform.CameraFromWorld(shutterOpen).GetMatrix();
}

void CameraBase::FindMinimumDifferentials(CameraHandle camera) {
    minPosDifferentialX = minPosDifferentialY = minDirDifferentialX =
        minDirDifferentialY = Vector3f(Infinity, Infinity, Infinity);

    CameraSample sample;
    sample.pLens = Point2f(0.5, 0.5);
    sample.time = 0.5;
    SampledWavelengths lambda = SampledWavelengths::SampleXYZ(0.5);

    int n = 512;
    for (int i = 0; i < n; ++i) {
        sample.pFilm.x = Float(i) / (n - 1) * film.FullResolution().x;
        sample.pFilm.y = Float(i) / (n - 1) * film.FullResolution().y;

        pstd::optional<CameraRayDifferential> crd =
            camera.GenerateRayDifferential(sample, lambda);
        if (!crd)
            continue;

        RayDifferential &ray = crd->ray;
        Vector3f dox = CameraFromRender(ray.rxOrigin - ray.o, ray.time);
        if (Length(dox) < Length(minPosDifferentialX))
            minPosDifferentialX = dox;
        Vector3f doy = CameraFromRender(ray.ryOrigin - ray.o, ray.time);
        if (Length(doy) < Length(minPosDifferentialY))
            minPosDifferentialY = doy;

        ray.d = Normalize(ray.d);
        ray.rxDirection = Normalize(ray.rxDirection);
        ray.ryDirection = Normalize(ray.ryDirection);

        Frame f = Frame::FromZ(ray.d);
        Vector3f df = f.ToLocal(ray.d);  // should be (0, 0, 1);
        Vector3f dxf = Normalize(f.ToLocal(ray.rxDirection));
        Vector3f dyf = Normalize(f.ToLocal(ray.ryDirection));

        if (Length(dxf - df) < Length(minDirDifferentialX))
            minDirDifferentialX = dxf - df;
        if (Length(dyf - df) < Length(minDirDifferentialY))
            minDirDifferentialY = dyf - df;
    }

    LOG_VERBOSE("Camera min pos differentials: %s, %s", minPosDifferentialX,
                minPosDifferentialY);
    LOG_VERBOSE("Camera min dir differentials: %s, %s", minDirDifferentialX,
                minDirDifferentialY);
}

std::string CameraBase::ToString() const {
    return StringPrintf("cameraTransform: %s shutterOpen: %f shutterClose: %f film: %s "
                        "medium: %s minPosDifferentialX: %s minPosDifferentialY: %s "
                        "minDirDifferentialX: %s minDirDifferentialY: %s ",
                        cameraTransform, shutterOpen, shutterClose, film,
                        medium ? medium.ToString().c_str() : "(nullptr)",
                        minPosDifferentialX, minPosDifferentialY, minDirDifferentialX,
                        minDirDifferentialY);
}

std::string CameraSample::ToString() const {
    return StringPrintf("[ pFilm: %s pLens: %s time: %f weight: %f ]", pFilm, pLens, time,
                        weight);
}

// ProjectiveCamera Method Definitions
void ProjectiveCamera::InitMetadata(ImageMetadata *metadata) const {
    metadata->cameraFromWorld = cameraTransform.CameraFromWorld(shutterOpen).GetMatrix();

    // TODO: double check this
    Transform NDCFromWorld = Translate(Vector3f(0.5, 0.5, 0.5)) * Scale(0.5, 0.5, 0.5) *
                             screenFromCamera * *metadata->cameraFromWorld;
    metadata->NDCFromWorld = NDCFromWorld.GetMatrix();

    CameraBase::InitMetadata(metadata);
}

std::string ProjectiveCamera::BaseToString() const {
    return CameraBase::ToString() +
           StringPrintf("screenFromCamera: %s cameraFromRaster: %s "
                        "rasterFromScreen: %s screenFromRaster: %s "
                        "lensRadius: %f focalDistance: %f",
                        screenFromCamera, cameraFromRaster, rasterFromScreen,
                        screenFromRaster, lensRadius, focalDistance);
}

CameraHandle CameraHandle::Create(const std::string &name,
                                  const ParameterDictionary &parameters,
                                  MediumHandle medium,
                                  const CameraTransform &cameraTransform, FilmHandle film,
                                  const FileLoc *loc, Allocator alloc) {
    CameraHandle camera;
    if (name == "perspective")
        camera = PerspectiveCamera::Create(parameters, cameraTransform, film, medium, loc,
                                           alloc);
    else if (name == "orthographic")
        camera = OrthographicCamera::Create(parameters, cameraTransform, film, medium,
                                            loc, alloc);
    else if (name == "realistic")
        camera = RealisticCamera::Create(parameters, cameraTransform, film, medium, loc,
                                         alloc);
    else if (name == "spherical")
        camera = SphericalCamera::Create(parameters, cameraTransform, film, medium, loc,
                                         alloc);
    else
        ErrorExit(loc, "%s: camera type unknown.", name);

    if (!camera)
        ErrorExit(loc, "%s: unable to create camera.", name);

    parameters.ReportUnused();
    return camera;
}

// CameraBaseParameters Method Definitions
CameraBaseParameters::CameraBaseParameters(const CameraTransform &cameraTransform,
                                           FilmHandle film, MediumHandle medium,
                                           const ParameterDictionary &parameters,
                                           const FileLoc *loc)
    : cameraTransform(cameraTransform), film(film), medium(medium) {
    shutterOpen = parameters.GetOneFloat("shutteropen", 0.f);
    shutterClose = parameters.GetOneFloat("shutterclose", 1.f);
    if (shutterClose < shutterOpen) {
        Warning(loc, "Shutter close time %f < shutter open %f.  Swapping them.",
                shutterClose, shutterOpen);
        pstd::swap(shutterClose, shutterOpen);
    }
}

// OrthographicCamera Method Definitions
pstd::optional<CameraRay> OrthographicCamera::GenerateRay(
    CameraSample sample, SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);

    Ray ray(pCamera, Vector3f(0, 0, 1), SampleTime(sample.time), medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }

    return CameraRay{RenderFromCamera(ray)};
}

pstd::optional<CameraRayDifferential> OrthographicCamera::GenerateRayDifferential(
    const CameraSample &sample, SampledWavelengths &lambda) const {
    // Compute main orthographic viewing ray
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);

    RayDifferential ray(pCamera, Vector3f(0, 0, 1), SampleTime(sample.time), medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }

    // Compute ray differentials for _OrthographicCamera_
    if (lensRadius > 0) {
        // Compute \use{OrthographicCamera} ray differentials accounting for lens
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = pCamera + dxCamera + (ft * Vector3f(0, 0, 1));
        ray.rxOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.rxDirection = Normalize(pFocus - ray.rxOrigin);

        pFocus = pCamera + dyCamera + (ft * Vector3f(0, 0, 1));
        ray.ryOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.ryDirection = Normalize(pFocus - ray.ryOrigin);

    } else {
        ray.rxOrigin = ray.o + dxCamera;
        ray.ryOrigin = ray.o + dyCamera;
        ray.rxDirection = ray.ryDirection = ray.d;
    }

    ray.hasDifferentials = true;
    return CameraRayDifferential{RenderFromCamera(ray)};
}

std::string OrthographicCamera::ToString() const {
    return StringPrintf("[ OrthographicCamera %s dxCamera: %s dyCamera: %s ]",
                        BaseToString(), dxCamera, dyCamera);
}

OrthographicCamera *OrthographicCamera::Create(const ParameterDictionary &parameters,
                                               const CameraTransform &cameraTransform,
                                               FilmHandle film, MediumHandle medium,
                                               const FileLoc *loc, Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    Float lensradius = parameters.GetOneFloat("lensradius", 0.f);
    Float focaldistance = parameters.GetOneFloat("focaldistance", 1e6f);
    Float frame =
        parameters.GetOneFloat("frameaspectratio", Float(film.FullResolution().x) /
                                                       Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = parameters.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (sw.size() == 4) {
            screen.pMin.x = sw[0];
            screen.pMax.x = sw[1];
            screen.pMin.y = sw[2];
            screen.pMax.y = sw[3];
        } else
            Error("\"screenwindow\" should have four values");
    }
    return alloc.new_object<OrthographicCamera>(cameraBaseParameters, screen, lensradius,
                                                focaldistance);
}

// PerspectiveCamera Method Definitions
pstd::optional<CameraRay> PerspectiveCamera::GenerateRay(
    CameraSample sample, SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);

    Ray ray(Point3f(0, 0, 0), Normalize(Vector3f(pCamera)), SampleTime(sample.time),
            medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }

    return CameraRay{RenderFromCamera(ray)};
}

pstd::optional<CameraRayDifferential> PerspectiveCamera::GenerateRayDifferential(
    const CameraSample &sample, SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);
    Vector3f dir = Normalize(Vector3f(pCamera.x, pCamera.y, pCamera.z));
    RayDifferential ray(Point3f(0, 0, 0), dir, SampleTime(sample.time), medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }

    // Compute offset rays for \use{PerspectiveCamera} ray differentials
    if (lensRadius > 0) {
        // Compute \use{PerspectiveCamera} ray differentials accounting for lens
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute $x$ ray differential for _PerspectiveCamera_ with lens
        Vector3f dx = Normalize(Vector3f(pCamera + dxCamera));
        Float ft = focalDistance / dx.z;
        Point3f pFocus = Point3f(0, 0, 0) + (ft * dx);
        ray.rxOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.rxDirection = Normalize(pFocus - ray.rxOrigin);

        // Compute $y$ ray differential for _PerspectiveCamera_ with lens
        Vector3f dy = Normalize(Vector3f(pCamera + dyCamera));
        ft = focalDistance / dy.z;
        pFocus = Point3f(0, 0, 0) + (ft * dy);
        ray.ryOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.ryDirection = Normalize(pFocus - ray.ryOrigin);

    } else {
        ray.rxOrigin = ray.ryOrigin = ray.o;
        ray.rxDirection = Normalize(Vector3f(pCamera) + dxCamera);
        ray.ryDirection = Normalize(Vector3f(pCamera) + dyCamera);
    }

    ray.hasDifferentials = true;
    return CameraRayDifferential{RenderFromCamera(ray)};
}

std::string PerspectiveCamera::ToString() const {
    return StringPrintf("[ PerspectiveCamera %s dxCamera: %s dyCamera: %s A: "
                        "%f cosTotalWidth: %f ]",
                        BaseToString(), dxCamera, dyCamera, A, cosTotalWidth);
}

PerspectiveCamera *PerspectiveCamera::Create(const ParameterDictionary &parameters,
                                             const CameraTransform &cameraTransform,
                                             FilmHandle film, MediumHandle medium,
                                             const FileLoc *loc, Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    Float lensradius = parameters.GetOneFloat("lensradius", 0.f);
    Float focaldistance = parameters.GetOneFloat("focaldistance", 1e6);
    Float frame =
        parameters.GetOneFloat("frameaspectratio", Float(film.FullResolution().x) /
                                                       Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = parameters.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (sw.size() == 4) {
            screen.pMin.x = sw[0];
            screen.pMax.x = sw[1];
            screen.pMin.y = sw[2];
            screen.pMax.y = sw[3];
        } else
            Error(loc, "\"screenwindow\" should have four values");
    }
    Float fov = parameters.GetOneFloat("fov", 90.);
    return alloc.new_object<PerspectiveCamera>(cameraBaseParameters, fov, screen,
                                               lensradius, focaldistance);
}

SampledSpectrum PerspectiveCamera::We(const Ray &ray, SampledWavelengths &lambda,
                                      Point2f *pRaster2) const {
    // Check if ray is forward-facing with respect to the camera
    Float cosTheta = Dot(ray.d, RenderFromCamera(Vector3f(0, 0, 1), ray.time));
    if (cosTheta <= cosTotalWidth)
        return SampledSpectrum(0.);

    // Map ray $(\p{}, \w{})$ onto the raster grid
    Point3f pFocus = ray((lensRadius > 0 ? focalDistance : 1) / cosTheta);
    Point3f pCamera = CameraFromRender(pFocus, ray.time);
    Point3f pRaster = cameraFromRaster.ApplyInverse(pCamera);

    // Return raster position if requested
    if (pRaster2)
        *pRaster2 = Point2f(pRaster.x, pRaster.y);

    // Return zero importance for out of bounds points
    Bounds2f sampleBounds = film.SampleBounds();
    if (!Inside(Point2f(pRaster.x, pRaster.y), sampleBounds))
        return SampledSpectrum(0.);

    // Compute lens area of perspective camera
    Float lensArea = lensRadius != 0 ? (Pi * lensRadius * lensRadius) : 1;

    // Return importance for point on image plane
    return SampledSpectrum(1 / (A * lensArea * Pow<4>(cosTheta)));
}

void PerspectiveCamera::PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    // Return zero PDF values if ray direction is not front-facing
    Float cosTheta = Dot(ray.d, RenderFromCamera(Vector3f(0, 0, 1), ray.time));
    if (cosTheta <= cosTotalWidth) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    // Map ray $(\p{}, \w{})$ onto the raster grid
    Point3f pFocus = ray((lensRadius > 0 ? focalDistance : 1) / cosTheta);
    Point3f pCamera = CameraFromRender(pFocus, ray.time);
    Point3f pRaster = cameraFromRaster.ApplyInverse(pCamera);

    // Return zero probability for out of bounds points
    Bounds2f sampleBounds = film.SampleBounds();
    if (!Inside(Point2f(pRaster.x, pRaster.y), sampleBounds)) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    // Compute lens area  and return perspective camera probabilities
    Float lensArea = lensRadius != 0 ? (Pi * lensRadius * lensRadius) : 1;
    *pdfPos = 1 / lensArea;
    *pdfDir = 1 / (A * Pow<3>(cosTheta));
}

pstd::optional<CameraWiSample> PerspectiveCamera::SampleWi(
    const Interaction &ref, const Point2f &u, SampledWavelengths &lambda) const {
    // Uniformly sample a lens interaction _lensIntr_
    Point2f pLens = lensRadius * SampleUniformDiskConcentric(u);
    Point3f pLensRender = RenderFromCamera(Point3f(pLens.x, pLens.y, 0), ref.time);
    Normal3f n = Normal3f(RenderFromCamera(Vector3f(0, 0, 1), ref.time));
    Interaction lensIntr(pLensRender, n, ref.time, medium);

    // Populate arguments and compute the importance value
    // Compute incident direction to camera _wi_ at _ref_
    Vector3f wi = lensIntr.p() - ref.p();
    Float dist = Length(wi);
    wi /= dist;

    // Compute PDF for importance arriving at _ref_
    Float lensArea = lensRadius != 0 ? (Pi * lensRadius * lensRadius) : 1;
    Float pdf = (dist * dist) / (AbsDot(lensIntr.n, wi) * lensArea);

    Point2f pRaster;
    SampledSpectrum Wi = We(lensIntr.SpawnRay(-wi), lambda, &pRaster);
    if (!Wi)
        return {};
    return CameraWiSample(Wi, wi, pdf, pRaster, ref, lensIntr);
}

// SphericalCamera Method Definitions
pstd::optional<CameraRay> SphericalCamera::GenerateRay(CameraSample sample,
                                                       SampledWavelengths &lambda) const {
    // Compute spherical camera ray direction
    Point2f uv(sample.pFilm.x / film.FullResolution().x,
               sample.pFilm.y / film.FullResolution().y);
    Vector3f dir;
    if (mapping == EquiRectangular) {
        // Compute ray direction using equirectangular mapping
        Float theta = Pi * uv[1], phi = 2 * Pi * uv[0];
        dir = SphericalDirection(std::sin(theta), std::cos(theta), phi);

    } else {
        // Compute ray direction using equal area mapping
        uv = WrapEqualAreaSquare(uv);
        dir = EqualAreaSquareToSphere(uv);
    }
    pstd::swap(dir.y, dir.z);

    Ray ray(Point3f(0, 0, 0), dir, SampleTime(sample.time), medium);
    return CameraRay{RenderFromCamera(ray)};
}

SphericalCamera *SphericalCamera::Create(const ParameterDictionary &parameters,
                                         const CameraTransform &cameraTransform,
                                         FilmHandle film, MediumHandle medium,
                                         const FileLoc *loc, Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    Float lensradius = parameters.GetOneFloat("lensradius", 0.f);
    Float focaldistance = parameters.GetOneFloat("focaldistance", 1e30f);
    Float frame =
        parameters.GetOneFloat("frameaspectratio", Float(film.FullResolution().x) /
                                                       Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = parameters.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (sw.size() == 4) {
            screen.pMin.x = sw[0];
            screen.pMax.x = sw[1];
            screen.pMin.y = sw[2];
            screen.pMax.y = sw[3];
        } else
            Error(loc, "\"screenwindow\" should have four values");
    }
    (void)lensradius;     // don't need this
    (void)focaldistance;  // don't need this

    std::string m = parameters.GetOneString("mapping", "equalarea");
    Mapping mapping;
    if (m == "equalarea")
        mapping = EqualArea;
    else if (m == "equirectangular")
        mapping = EquiRectangular;
    else
        ErrorExit(loc,
                  "%s: unknown mapping for spherical camera. (Must be "
                  "\"equalarea\" or \"equirectangular\".)",
                  m);

    return alloc.new_object<SphericalCamera>(cameraBaseParameters, mapping);
}

std::string SphericalCamera::ToString() const {
    return StringPrintf("[ SphericalCamera %s mapping: %s ]", CameraBase::ToString(),
                        mapping == EquiRectangular ? "EquiRectangular" : "EqualArea");
}

// RealisticCamera Method Definitions
RealisticCamera::RealisticCamera(CameraBaseParameters baseParameters,
                                 std::vector<Float> &lensParameters, Float focusDistance,
                                 Float setApertureDiameter, Image apertureImage,
                                 Allocator alloc)
    : CameraBase(baseParameters),
      elementInterfaces(alloc),
      exitPupilBounds(alloc),
      apertureImage(std::move(apertureImage)) {
    // Compute film's physical extent
    Float aspect = (Float)film.FullResolution().y / (Float)film.FullResolution().x;
    Float diagonal = film.Diagonal();
    Float x = std::sqrt(diagonal * diagonal / (1 + aspect * aspect));
    Float y = aspect * x;
    physicalExtent = Bounds2f(Point2f(-x / 2, -y / 2), Point2f(x / 2, y / 2));

    // Initialize _elementInterfaces_ for camera
    for (size_t i = 0; i < lensParameters.size(); i += 4) {
        // Extract lens element configuration from _lensParameters_
        Float curvatureRadius = lensParameters[i] / 1000;
        Float thickness = lensParameters[i + 1] / 1000;
        Float eta = lensParameters[i + 2];
        Float apertureDiameter = lensParameters[i + 3] / 1000;

        if (curvatureRadius == 0) {
            // Set aperture stop diameter
            setApertureDiameter /= 1000;
            if (setApertureDiameter > apertureDiameter)
                Warning("Aperture diameter %f is greater than maximum possible %f. "
                        "Clamping it.",
                        setApertureDiameter, apertureDiameter);
            else
                apertureDiameter = setApertureDiameter;
        }
        // Add element interface to end of _elementInterfaces_
        elementInterfaces.push_back(
            {curvatureRadius, thickness, eta, apertureDiameter / 2});
    }

    // Compute lens--film distance for given focus distance
    elementInterfaces.back().thickness = FocusThickLens(focusDistance);

    // Compute exit pupil bounds at sampled points on the film
    int nSamples = 64;
    exitPupilBounds.resize(nSamples);
    ParallelFor(0, nSamples, [&](int i) {
        Float r0 = (Float)i / nSamples * film.Diagonal() / 2;
        Float r1 = (Float)(i + 1) / nSamples * film.Diagonal() / 2;
        exitPupilBounds[i] = BoundExitPupil(r0, r1);
    });

    // Compute minimum differentials for _RealisticCamera_
    FindMinimumDifferentials(this);
}

Float RealisticCamera::TraceLensesFromFilm(const Ray &rCamera, Ray *rOut) const {
    Float elementZ = 0, weight = 1;
    // Transform _rCamera_ from camera to lens system space
    Ray rLens(Point3f(rCamera.o.x, rCamera.o.y, -rCamera.o.z),
              Vector3f(rCamera.d.x, rCamera.d.y, -rCamera.d.z), rCamera.time);

    for (int i = elementInterfaces.size() - 1; i >= 0; --i) {
        const LensElementInterface &element = elementInterfaces[i];
        // Update ray from film accounting for interaction with _element_
        elementZ -= element.thickness;
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        bool isStop = (element.curvatureRadius == 0);
        if (isStop) {
            // Compute _t_ at plane of aperture stop
            t = (elementZ - rLens.o.z) / rLens.d.z;
            if (t < 0)
                return 0;

        } else {
            // Intersect ray with element to compute _t_ and _n_
            Float radius = element.curvatureRadius;
            Float zCenter = elementZ + element.curvatureRadius;
            if (!IntersectSphericalElement(radius, zCenter, rLens, &t, &n))
                return 0;
        }
        DCHECK_GE(t, 0);

        // Test intersection point against element aperture
        Point3f pHit = rLens(t);
        if (isStop && apertureImage) {
            // Check intersection point against _apertureImage_
            Point2f uv((pHit.x / element.apertureRadius + 1) / 2,
                       (pHit.y / element.apertureRadius + 1) / 2);
            uv.y = 1 - uv.y;
            weight = apertureImage.BilerpChannel(uv, 0, WrapMode::Black);
            if (weight == 0)
                return 0;

        } else {
            // Check intersection point against spherical aperture
            Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
            if (r2 > element.apertureRadius * element.apertureRadius)
                return 0;
        }
        rLens.o = pHit;

        // Update ray path for element interface interaction
        if (!isStop) {
            Vector3f w;
            Float eta_i = element.eta;
            Float eta_t = (i > 0 && elementInterfaces[i - 1].eta != 0)
                              ? elementInterfaces[i - 1].eta
                              : 1;
            if (!Refract(Normalize(-rLens.d), n, eta_t / eta_i, &w))
                return 0;
            rLens.d = w;
        }
    }
    // Transform _rLens_ from lens system space back to camera space
    if (rOut != nullptr)
        *rOut = Ray(Point3f(rLens.o.x, rLens.o.y, -rLens.o.z),
                    Vector3f(rLens.d.x, rLens.d.y, -rLens.d.z), rLens.time);

    return weight;
}

void RealisticCamera::ComputeCardinalPoints(const Ray &rIn, const Ray &rOut, Float *pz,
                                            Float *fz) {
    Float tf = -rOut.o.x / rOut.d.x;
    *fz = -rOut(tf).z;
    Float tp = (rIn.o.x - rOut.o.x) / rOut.d.x;
    *pz = -rOut(tp).z;
}

void RealisticCamera::ComputeThickLensApproximation(Float pz[2], Float fz[2]) const {
    // Find height $x$ from optical axis for parallel rays
    Float x = .001f * film.Diagonal();

    // Compute cardinal points for film side of lens system
    Ray rScene(Point3f(x, 0, LensFrontZ() + 1), Vector3f(0, 0, -1));
    Ray rFilm;
    if (!TraceLensesFromScene(rScene, &rFilm))
        ErrorExit("Unable to trace ray from scene to film for thick lens "
                  "approximation. Is aperture stop extremely small?");
    ComputeCardinalPoints(rScene, rFilm, &pz[0], &fz[0]);

    // Compute cardinal points for scene side of lens system
    rFilm = Ray(Point3f(x, 0, LensRearZ() - 1), Vector3f(0, 0, 1));
    if (TraceLensesFromFilm(rFilm, &rScene) == 0)
        ErrorExit("Unable to trace ray from film to scene for thick lens "
                  "approximation. Is aperture stop extremely small?");
    ComputeCardinalPoints(rFilm, rScene, &pz[1], &fz[1]);
}

Float RealisticCamera::FocusThickLens(Float focusDistance) {
    Float pz[2], fz[2];
    ComputeThickLensApproximation(pz, fz);
    LOG_VERBOSE("Cardinal points: p' = %f f' = %f, p = %f f = %f.\n", pz[0], fz[0], pz[1],
                fz[1]);
    LOG_VERBOSE("Effective focal length %f\n", fz[0] - pz[0]);
    // Compute translation of lens, _delta_, to focus at _focusDistance_
    Float f = fz[0] - pz[0];
    Float z = -focusDistance;
    Float c = (pz[1] - z - pz[0]) * (pz[1] - z - 4 * f - pz[0]);
    if (c <= 0)
        ErrorExit("Coefficient must be positive. It looks focusDistance %f "
                  " is too short for a given lenses configuration",
                  focusDistance);
    Float delta = (pz[1] - z + pz[0] - std::sqrt(c)) / 2;

    return elementInterfaces.back().thickness + delta;
}

Bounds2f RealisticCamera::BoundExitPupil(Float filmX0, Float filmX1) const {
    Bounds2f pupilBounds;
    // Sample a collection of points on the rear lens to find exit pupil
    const int nSamples = 1024 * 1024;
    int nExitingRays = 0;
    // Compute bounding box of projection of rear element on sampling plane
    Float rearRadius = RearElementRadius();
    Bounds2f projRearBounds(Point2f(-1.5f * rearRadius, -1.5f * rearRadius),
                            Point2f(1.5f * rearRadius, 1.5f * rearRadius));

    for (int i = 0; i < nSamples; ++i) {
        // Find location of sample points on $x$ segment and rear lens element
        Point3f pFilm(Lerp((i + 0.5f) / nSamples, filmX0, filmX1), 0, 0);
        Float u[2] = {RadicalInverse(0, i), RadicalInverse(1, i)};
        Point3f pRear(Lerp(u[0], projRearBounds.pMin.x, projRearBounds.pMax.x),
                      Lerp(u[1], projRearBounds.pMin.y, projRearBounds.pMax.y),
                      LensRearZ());

        // Expand pupil bounds if ray makes it through the lens system
        if (Inside(Point2f(pRear.x, pRear.y), pupilBounds) ||
            TraceLensesFromFilm(Ray(pFilm, pRear - pFilm), nullptr)) {
            pupilBounds = Union(pupilBounds, Point2f(pRear.x, pRear.y));
            ++nExitingRays;
        }
    }

    // Return entire element bounds if no rays made it through the lens system
    if (nExitingRays == 0) {
        LOG_VERBOSE("Unable to find exit pupil in x = [%f,%f] on film.", filmX0, filmX1);
        return projRearBounds;
    }

    // Expand bounds to account for sample spacing
    pupilBounds =
        Expand(pupilBounds, 2 * Length(projRearBounds.Diagonal()) / std::sqrt(nSamples));

    return pupilBounds;
}

Point3f RealisticCamera::SampleExitPupil(const Point2f &pFilm, const Point2f &lensSample,
                                         Float *sampleBoundsArea) const {
    // Find exit pupil bound for sample distance from film center
    Float rFilm = std::sqrt(pFilm.x * pFilm.x + pFilm.y * pFilm.y);
    int rIndex = rFilm / (film.Diagonal() / 2) * exitPupilBounds.size();
    rIndex = std::min<int>(exitPupilBounds.size() - 1, rIndex);
    Bounds2f pupilBounds = exitPupilBounds[rIndex];
    if (sampleBoundsArea != nullptr)
        *sampleBoundsArea = pupilBounds.Area();

    // Generate sample point inside exit pupil bound
    Point2f pLens = pupilBounds.Lerp(lensSample);

    // Return sample point rotated by angle of _pFilm_ with $+x$ axis
    Float sinTheta = (rFilm != 0) ? pFilm.y / rFilm : 0;
    Float cosTheta = (rFilm != 0) ? pFilm.x / rFilm : 1;
    return {cosTheta * pLens.x - sinTheta * pLens.y,
            sinTheta * pLens.x + cosTheta * pLens.y, LensRearZ()};
}

pstd::optional<CameraRay> RealisticCamera::GenerateRay(CameraSample sample,
                                                       SampledWavelengths &lambda) const {
    // Find point on film, _pFilm_, corresponding to _sample.pFilm_
    Point2f s(sample.pFilm.x / film.FullResolution().x,
              sample.pFilm.y / film.FullResolution().y);
    Point2f pFilm2 = physicalExtent.Lerp(s);
    Point3f pFilm(-pFilm2.x, pFilm2.y, 0);

    // Trace ray from _pFilm_ through lens system
    Float exitPupilBoundsArea;
    Point3f pRear =
        SampleExitPupil(Point2f(pFilm.x, pFilm.y), sample.pLens, &exitPupilBoundsArea);
    Ray rFilm(pFilm, pRear - pFilm);
    Ray ray;
    Float weight = TraceLensesFromFilm(rFilm, &ray);
    if (weight == 0)
        return {};

    // Finish initialization of _RealisticCamera_ ray
    ray.time = SampleTime(sample.time);
    ray.medium = medium;
    ray = RenderFromCamera(ray);
    ray.d = Normalize(ray.d);

    // Compute weighting for _RealisticCamera_ ray
    Float cosTheta = Normalize(rFilm.d).z;
    weight *= (shutterClose - shutterOpen) * Pow<4>(cosTheta) * exitPupilBoundsArea /
              Sqr(LensRearZ());

    return CameraRay{ray, SampledSpectrum(weight)};
}

STAT_PERCENT("Camera/Rays vignetted by lens system", vignettedRays, totalRays);

std::string RealisticCamera::LensElementInterface::ToString() const {
    return StringPrintf("[ LensElementInterface curvatureRadius: %f thickness: %f "
                        "eta: %f apertureRadius: %f ]",
                        curvatureRadius, thickness, eta, apertureRadius);
}

Float RealisticCamera::TraceLensesFromScene(const Ray &rCamera, Ray *rOut) const {
    Float elementZ = -LensFrontZ();
    // Transform _rCamera_ from camera to lens system space
    const Transform LensFromCamera = Scale(1, 1, -1);
    Ray rLens = LensFromCamera(rCamera);
    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        bool isStop = (element.curvatureRadius == 0);
        if (isStop) {
            t = (elementZ - rLens.o.z) / rLens.d.z;
            if (t < 0)
                return 0;
        } else {
            Float radius = element.curvatureRadius;
            Float zCenter = elementZ + element.curvatureRadius;
            if (!IntersectSphericalElement(radius, zCenter, rLens, &t, &n))
                return 0;
        }

        // Test intersection point against element aperture
        // Don't worry about the aperture image here.
        Point3f pHit = rLens(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        if (r2 > element.apertureRadius * element.apertureRadius)
            return 0;
        rLens.o = pHit;

        // Update ray path for from-scene element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = (i == 0 || elementInterfaces[i - 1].eta == 0)
                              ? 1
                              : elementInterfaces[i - 1].eta;
            Float eta_t = (elementInterfaces[i].eta != 0) ? elementInterfaces[i].eta : 1;
            if (!Refract(Normalize(-rLens.d), n, eta_t / eta_i, &wt))
                return 0;
            rLens.d = wt;
        }
        elementZ += element.thickness;
    }
    // Transform _rLens_ from lens system space back to camera space
    if (rOut != nullptr)
        *rOut = Ray(Point3f(rLens.o.x, rLens.o.y, -rLens.o.z),
                    Vector3f(rLens.d.x, rLens.d.y, -rLens.d.z), rLens.time);
    return 1;
}

void RealisticCamera::DrawLensSystem() const {
    Float sumz = -LensFrontZ();
    Float z = sumz;
    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        Float r = element.curvatureRadius;
        if (r == 0) {
            // stop
            printf("{Thick, Line[{{%f, %f}, {%f, %f}}], ", z, element.apertureRadius, z,
                   2 * element.apertureRadius);
            printf("Line[{{%f, %f}, {%f, %f}}]}, ", z, -element.apertureRadius, z,
                   -2 * element.apertureRadius);
        } else {
            Float theta = std::abs(SafeASin(element.apertureRadius / r));
            if (r > 0) {
                // convex as seen from front of lens
                Float t0 = Pi - theta;
                Float t1 = Pi + theta;
                printf("Circle[{%f, 0}, %f, {%f, %f}], ", z + r, r, t0, t1);
            } else {
                // concave as seen from front of lens
                Float t0 = -theta;
                Float t1 = theta;
                printf("Circle[{%f, 0}, %f, {%f, %f}], ", z + r, -r, t0, t1);
            }
            if (element.eta != 0 && element.eta != 1) {
                // connect top/bottom to next element
                CHECK_LT(i + 1, elementInterfaces.size());
                Float nextApertureRadius = elementInterfaces[i + 1].apertureRadius;
                Float h = std::max(element.apertureRadius, nextApertureRadius);
                Float hlow = std::min(element.apertureRadius, nextApertureRadius);

                Float zp0, zp1;
                if (r > 0) {
                    zp0 = z + element.curvatureRadius -
                          element.apertureRadius / std::tan(theta);
                } else {
                    zp0 = z + element.curvatureRadius +
                          element.apertureRadius / std::tan(theta);
                }

                Float nextCurvatureRadius = elementInterfaces[i + 1].curvatureRadius;
                Float nextTheta =
                    std::abs(SafeASin(nextApertureRadius / nextCurvatureRadius));
                if (nextCurvatureRadius > 0) {
                    zp1 = z + element.thickness + nextCurvatureRadius -
                          nextApertureRadius / std::tan(nextTheta);
                } else {
                    zp1 = z + element.thickness + nextCurvatureRadius +
                          nextApertureRadius / std::tan(nextTheta);
                }

                // Connect tops
                printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, h, zp1, h);
                printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, -h, zp1, -h);

                // vertical lines when needed to close up the element profile
                if (element.apertureRadius < nextApertureRadius) {
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, h, zp0, hlow);
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, -h, zp0, -hlow);
                } else if (element.apertureRadius > nextApertureRadius) {
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp1, h, zp1, hlow);
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp1, -h, zp1, -hlow);
                }
            }
        }
        z += element.thickness;
    }

    // 24mm height for 35mm film
    printf("Line[{{0, -.012}, {0, .012}}], ");
    // optical axis
    printf("Line[{{0, 0}, {%f, 0}}] ", 1.2f * sumz);
}

void RealisticCamera::DrawRayPathFromFilm(const Ray &r, bool arrow,
                                          bool toOpticalIntercept) const {
    Float elementZ = 0;
    // Transform _ray_ from camera to lens system space
    static const Transform LensFromCamera = Scale(1, 1, -1);
    Ray ray = LensFromCamera(r);
    printf("{ ");
    if (TraceLensesFromFilm(r, nullptr) == 0) {
        printf("Dashed, RGBColor[.8, .5, .5]");
    } else
        printf("RGBColor[.5, .5, .8]");

    for (int i = elementInterfaces.size() - 1; i >= 0; --i) {
        const LensElementInterface &element = elementInterfaces[i];
        elementZ -= element.thickness;
        bool isStop = (element.curvatureRadius == 0);
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        if (isStop)
            t = -(ray.o.z - elementZ) / ray.d.z;
        else {
            Float radius = element.curvatureRadius;
            Float zCenter = elementZ + element.curvatureRadius;
            if (!IntersectSphericalElement(radius, zCenter, ray, &t, &n))
                goto done;
        }
        CHECK_GE(t, 0);

        printf(", Line[{{%f, %f}, {%f, %f}}]", ray.o.z, ray.o.x, ray(t).z, ray(t).x);

        // Test intersection point against element aperture
        Point3f pHit = ray(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        Float apertureRadius2 = element.apertureRadius * element.apertureRadius;
        if (r2 > apertureRadius2)
            goto done;
        ray.o = pHit;

        // Update ray path for element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = element.eta;
            Float eta_t = (i > 0 && elementInterfaces[i - 1].eta != 0)
                              ? elementInterfaces[i - 1].eta
                              : 1;
            if (!Refract(Normalize(-ray.d), n, eta_t / eta_i, &wt))
                goto done;
            ray.d = wt;
        }
    }

    ray.d = Normalize(ray.d);
    {
        Float ta = std::abs(elementZ / 4);
        if (toOpticalIntercept) {
            ta = -ray.o.x / ray.d.x;
            printf(", Point[{%f, %f}]", ray(ta).z, ray(ta).x);
        }
        printf(", %s[{{%f, %f}, {%f, %f}}]", arrow ? "Arrow" : "Line", ray.o.z, ray.o.x,
               ray(ta).z, ray(ta).x);

        // overdraw the optical axis if needed...
        if (toOpticalIntercept)
            printf(", Line[{{%f, 0}, {%f, 0}}]", ray.o.z, ray(ta).z * 1.05f);
    }

done:
    printf("}");
}

void RealisticCamera::DrawRayPathFromScene(const Ray &r, bool arrow,
                                           bool toOpticalIntercept) const {
    Float elementZ = LensFrontZ() * -1;

    // Transform _ray_ from camera to lens system space
    static const Transform LensFromCamera = Scale(1, 1, -1);
    Ray ray = LensFromCamera(r);
    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        bool isStop = (element.curvatureRadius == 0);
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        if (isStop)
            t = -(ray.o.z - elementZ) / ray.d.z;
        else {
            Float radius = element.curvatureRadius;
            Float zCenter = elementZ + element.curvatureRadius;
            if (!IntersectSphericalElement(radius, zCenter, ray, &t, &n))
                return;
        }
        CHECK_GE(t, 0.f);

        printf("Line[{{%f, %f}, {%f, %f}}],", ray.o.z, ray.o.x, ray(t).z, ray(t).x);

        // Test intersection point against element aperture
        Point3f pHit = ray(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        Float apertureRadius2 = element.apertureRadius * element.apertureRadius;
        if (r2 > apertureRadius2)
            return;
        ray.o = pHit;

        // Update ray path for from-scene element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = (i == 0 || elementInterfaces[i - 1].eta == 0.f)
                              ? 1.f
                              : elementInterfaces[i - 1].eta;
            Float eta_t =
                (elementInterfaces[i].eta != 0.f) ? elementInterfaces[i].eta : 1.f;
            if (!Refract(Normalize(-ray.d), n, eta_t / eta_i, &wt))
                return;
            ray.d = wt;
        }
        elementZ += element.thickness;
    }

    // go to the film plane by default
    {
        Float ta = -ray.o.z / ray.d.z;
        if (toOpticalIntercept) {
            ta = -ray.o.x / ray.d.x;
            printf("Point[{%f, %f}], ", ray(ta).z, ray(ta).x);
        }
        printf("%s[{{%f, %f}, {%f, %f}}]", arrow ? "Arrow" : "Line", ray.o.z, ray.o.x,
               ray(ta).z, ray(ta).x);
    }
}

void RealisticCamera::RenderExitPupil(Float sx, Float sy, const char *filename) const {
    Point3f pFilm(sx, sy, 0);

    const int nSamples = 2048;
    Image image(PixelFormat::Float, {nSamples, nSamples}, {"Y"});

    for (int y = 0; y < nSamples; ++y) {
        Float fy = (Float)y / (Float)(nSamples - 1);
        Float ly = Lerp(fy, -RearElementRadius(), RearElementRadius());
        for (int x = 0; x < nSamples; ++x) {
            Float fx = (Float)x / (Float)(nSamples - 1);
            Float lx = Lerp(fx, -RearElementRadius(), RearElementRadius());

            Point3f pRear(lx, ly, LensRearZ());

            if (lx * lx + ly * ly > RearElementRadius() * RearElementRadius())
                image.SetChannel({x, y}, 0, 1.);
            else if (TraceLensesFromFilm(Ray(pFilm, pRear - pFilm), nullptr))
                image.SetChannel({x, y}, 0, 0.5);
            else
                image.SetChannel({x, y}, 0, 0.);
        }
    }

    image.Write(filename);
}

void RealisticCamera::TestExitPupilBounds() const {
    Float filmDiagonal = film.Diagonal();

    static RNG rng;

    Float u = rng.Uniform<Float>();
    Point3f pFilm(u * filmDiagonal / 2, 0, 0);

    Float r = pFilm.x / (filmDiagonal / 2);
    int pupilIndex = std::min<int>(exitPupilBounds.size() - 1,
                                   std::floor(r * (exitPupilBounds.size() - 1)));
    Bounds2f pupilBounds = exitPupilBounds[pupilIndex];
    if (pupilIndex + 1 < (int)exitPupilBounds.size())
        pupilBounds = Union(pupilBounds, exitPupilBounds[pupilIndex + 1]);

    // Now, randomly pick points on the aperture and see if any are outside
    // of pupil bounds...
    for (int i = 0; i < 1000; ++i) {
        Point2f u2{rng.Uniform<Float>(), rng.Uniform<Float>()};
        Point2f pd = SampleUniformDiskConcentric(u2);
        pd *= RearElementRadius();

        Ray testRay(pFilm, Point3f(pd.x, pd.y, 0.f) - pFilm);
        Ray testOut;
        if (!TraceLensesFromFilm(testRay, &testOut))
            continue;

        if (!Inside(pd, pupilBounds)) {
            fprintf(stderr,
                    "Aha! (%f,%f) went through, but outside bounds (%f,%f) - "
                    "(%f,%f)\n",
                    pd.x, pd.y, pupilBounds.pMin[0], pupilBounds.pMin[1],
                    pupilBounds.pMax[0], pupilBounds.pMax[1]);
            RenderExitPupil(
                (Float)pupilIndex / exitPupilBounds.size() * filmDiagonal / 2.f, 0.f,
                "low.exr");
            RenderExitPupil(
                (Float)(pupilIndex + 1) / exitPupilBounds.size() * filmDiagonal / 2.f,
                0.f, "high.exr");
            RenderExitPupil(pFilm.x, 0.f, "mid.exr");
            exit(0);
        }
    }
    fprintf(stderr, ".");
}

std::string RealisticCamera::ToString() const {
    return StringPrintf(
        "[ RealisticCamera %s elementInterfaces: %s exitPupilBounds: %s ]",
        CameraBase::ToString(), elementInterfaces, exitPupilBounds);
}

RealisticCamera *RealisticCamera::Create(const ParameterDictionary &parameters,
                                         const CameraTransform &cameraTransform,
                                         FilmHandle film, MediumHandle medium,
                                         const FileLoc *loc, Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    // Realistic camera-specific parameters
    std::string lensFile = ResolveFilename(parameters.GetOneString("lensfile", ""));
    Float apertureDiameter = parameters.GetOneFloat("aperturediameter", 1.0);
    Float focusDistance = parameters.GetOneFloat("focusdistance", 10.0);

    if (lensFile.empty()) {
        Error(loc, "No lens description file supplied!");
        return nullptr;
    }
    // Load element data from lens description file
    std::vector<Float> lensParameters = ReadFloatFile(lensFile);
    if (lensParameters.empty()) {
        Error(loc, "Error reading lens specification file \"%s\".", lensFile);
        return nullptr;
    }
    if (lensParameters.size() % 4 != 0) {
        Error(loc,
              "%s: excess values in lens specification file; "
              "must be multiple-of-four values, read %d.",
              lensFile, (int)lensParameters.size());
        return nullptr;
    }

    int builtinRes = 256;
    auto rasterize = [&](pstd::span<const Point2f> vert) {
        Image image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"}, nullptr, alloc);

        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                Point2f p(-1 + 2 * (x + 0.5f) / image.Resolution().x,
                          -1 + 2 * (y + 0.5f) / image.Resolution().y);
                int windingNumber = 0;
                // Test against edges
                for (int i = 0; i < vert.size(); ++i) {
                    int i1 = (i + 1) % vert.size();
                    Float e = (p[0] - vert[i][0]) * (vert[i1][1] - vert[i][1]) -
                              (p[1] - vert[i][1]) * (vert[i1][0] - vert[i][0]);
                    if (vert[i].y <= p.y) {
                        if (vert[i1].y > p.y && e > 0)
                            ++windingNumber;
                    } else if (vert[i1].y <= p.y && e < 0)
                        --windingNumber;
                }

                image.SetChannel({x, y}, 0, windingNumber == 0 ? 0.f : 1.f);
            }

        return image;
    };

    std::string apertureName = ResolveFilename(parameters.GetOneString("aperture", ""));
    Image apertureImage;
    if (!apertureName.empty()) {
        // built-in diaphragm shapes
        if (apertureName == "gaussian") {
            apertureImage = Image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"},
                                  nullptr, alloc);
            for (int y = 0; y < apertureImage.Resolution().y; ++y)
                for (int x = 0; x < apertureImage.Resolution().x; ++x) {
                    Point2f uv(-1 + 2 * (x + 0.5f) / apertureImage.Resolution().x,
                               -1 + 2 * (y + 0.5f) / apertureImage.Resolution().y);
                    Float r2 = Sqr(uv.x) + Sqr(uv.y);
                    Float sigma2 = 1;
                    Float v = std::max<Float>(
                        0, std::exp(-r2 / sigma2) - std::exp(-1 / sigma2));
                    apertureImage.SetChannel({x, y}, 0, v);
                }
        } else if (apertureName == "square") {
            apertureImage = Image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"},
                                  nullptr, alloc);
            for (int y = 0; y < apertureImage.Resolution().y; ++y)
                for (int x = 0; x < apertureImage.Resolution().x; ++x)
                    apertureImage.SetChannel({x, y}, 0, 1.f);
        } else if (apertureName == "pentagon") {
            // https://mathworld.wolfram.com/RegularPentagon.html
            Float c1 = (std::sqrt(5.f) - 1) / 4;
            Float c2 = (std::sqrt(5.f) + 1) / 4;
            Float s1 = std::sqrt(10.f + 2.f * std::sqrt(5.f)) / 4;
            Float s2 = std::sqrt(10.f - 2.f * std::sqrt(5.f)) / 4;
            // Vertices in CW order.
            Point2f vert[5] = {Point2f(0, 1), {s1, c1}, {s2, -c2}, {-s2, -c2}, {-s1, c1}};
            // Scale down slightly
            for (int i = 0; i < 5; ++i)
                vert[i] *= .8f;
            apertureImage = rasterize(vert);
        } else if (apertureName == "star") {
            // 5-sided. Vertices are two pentagons--inner and outer radius
            pstd::array<Point2f, 10> vert;
            for (int i = 0; i < 10; ++i) {
                // inner radius: https://math.stackexchange.com/a/2136996
                Float r =
                    (i & 1) ? 1.f : (std::cos(Radians(72.f)) / std::cos(Radians(36.f)));
                vert[i] = Point2f(r * std::cos(Pi * i / 5.f), r * std::sin(Pi * i / 5.f));
            }
            std::reverse(vert.begin(), vert.end());
            apertureImage = rasterize(vert);
        } else {
            ImageAndMetadata im = Image::Read(apertureName, alloc);
            apertureImage = std::move(im.image);
            if (apertureImage.NChannels() > 1) {
                ImageChannelDesc rgbDesc = apertureImage.GetChannelDesc({"R", "G", "B"});
                if (!rgbDesc)
                    ErrorExit("%s: didn't find R, G, B channels to average for "
                              "aperture image.",
                              apertureName);

                Image mono(PixelFormat::Float, apertureImage.Resolution(), {"Y"}, nullptr,
                           alloc);
                for (int y = 0; y < mono.Resolution().y; ++y)
                    for (int x = 0; x < mono.Resolution().x; ++x) {
                        Float avg = apertureImage.GetChannels({x, y}, rgbDesc).Average();
                        mono.SetChannel({x, y}, 0, avg);
                    }

                apertureImage = std::move(mono);
            }
        }

        if (apertureImage) {
            // Normalize it so that brightness matches a circular aperture
            Float sum = 0;
            for (int y = 0; y < apertureImage.Resolution().y; ++y)
                for (int x = 0; x < apertureImage.Resolution().x; ++x)
                    sum += apertureImage.GetChannel({x, y}, 0);
            Float avg =
                sum / (apertureImage.Resolution().x * apertureImage.Resolution().y);

            Float scale = (Pi / 4) / avg;
            for (int y = 0; y < apertureImage.Resolution().y; ++y)
                for (int x = 0; x < apertureImage.Resolution().x; ++x)
                    apertureImage.SetChannel({x, y}, 0,
                                             apertureImage.GetChannel({x, y}, 0) * scale);
        }
    }

    return alloc.new_object<RealisticCamera>(cameraBaseParameters, lensParameters,
                                             focusDistance, apertureDiameter,
                                             std::move(apertureImage), alloc);
}

}  // namespace pbrt
