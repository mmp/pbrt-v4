// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_PARSEDSCENE_H
#define PBRT_PARSEDSCENE_H

#include <pbrt/pbrt.h>

#include <pbrt/cameras.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/transform.h>

#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace pbrt {

// SceneEntity Definition
struct SceneEntity {
    // SceneEntity Public Methods
    SceneEntity() = default;
    SceneEntity(const std::string &name, ParameterDictionary parameters, FileLoc loc)
        : name(name), parameters(parameters), loc(loc) {}

    std::string ToString() const {
        return StringPrintf("[ SceneEntity name: %s parameters: %s loc: %s ]", name,
                            parameters, loc);
    }

    std::string name;
    FileLoc loc;
    ParameterDictionary parameters;
};

struct TransformedSceneEntity : public SceneEntity {
    TransformedSceneEntity() = default;
    TransformedSceneEntity(const std::string &name, ParameterDictionary parameters,
                           FileLoc loc, const AnimatedTransform &renderFromObject)
        : SceneEntity(name, parameters, loc), renderFromObject(renderFromObject) {}

    std::string ToString() const {
        return StringPrintf("[ TransformedSeneEntity name: %s parameters: %s loc: %s "
                            "renderFromObject: %s ]",
                            name, parameters, loc, renderFromObject);
    }

    AnimatedTransform renderFromObject;
};

// CameraSceneEntity Definition
struct CameraSceneEntity : public SceneEntity {
    // CameraSceneEntity Public Methods
    CameraSceneEntity() = default;
    CameraSceneEntity(const std::string &name, ParameterDictionary parameters,
                      FileLoc loc, const CameraTransform &cameraTransform,
                      const std::string &medium)
        : SceneEntity(name, parameters, loc),
          cameraTransform(cameraTransform),
          medium(medium) {}

    std::string ToString() const {
        return StringPrintf("[ CameraSeneEntity name: %s parameters: %s loc: %s "
                            "cameraTransform: %s medium: %s ]",
                            name, parameters, loc, cameraTransform, medium);
    }

    CameraTransform cameraTransform;
    std::string medium;
};

struct ShapeSceneEntity : public SceneEntity {
    ShapeSceneEntity() = default;
    ShapeSceneEntity(const std::string &name, ParameterDictionary parameters, FileLoc loc,
                     const Transform *renderFromObject, const Transform *objectFromRender,
                     bool reverseOrientation, int materialIndex,
                     const std::string &materialName, int lightIndex,
                     const std::string &insideMedium, const std::string &outsideMedium)
        : SceneEntity(name, parameters, loc),
          renderFromObject(renderFromObject),
          objectFromRender(objectFromRender),
          reverseOrientation(reverseOrientation),
          materialIndex(materialIndex),
          materialName(materialName),
          lightIndex(lightIndex),
          insideMedium(insideMedium),
          outsideMedium(outsideMedium) {}

    std::string ToString() const {
        return StringPrintf(
            "[ ShapeSeneEntity name: %s parameters: %s loc: %s "
            "renderFromObject: %s objectFromRender: %s reverseOrientation: %s "
            "materialIndex: %d materialName: %s lightIndex: %d "
            "insideMedium: %s outsideMedium: %s]",
            name, parameters, loc, *renderFromObject, *objectFromRender,
            reverseOrientation, materialIndex, materialName, lightIndex, insideMedium,
            outsideMedium);
    }

    const Transform *renderFromObject = nullptr, *objectFromRender = nullptr;
    bool reverseOrientation = false;
    int materialIndex;  // one of these two...  std::variant?
    std::string materialName;
    int lightIndex = -1;
    std::string insideMedium, outsideMedium;
};

struct AnimatedShapeSceneEntity : public TransformedSceneEntity {
    AnimatedShapeSceneEntity() = default;
    AnimatedShapeSceneEntity(const std::string &name, ParameterDictionary parameters,
                             FileLoc loc, const AnimatedTransform &renderFromObject,
                             const Transform *identity, bool reverseOrientation,
                             int materialIndex, const std::string &materialName,
                             int lightIndex, const std::string &insideMedium,
                             const std::string &outsideMedium)
        : TransformedSceneEntity(name, parameters, loc, renderFromObject),
          identity(identity),
          reverseOrientation(reverseOrientation),
          materialIndex(materialIndex),
          materialName(materialName),
          lightIndex(lightIndex),
          insideMedium(insideMedium),
          outsideMedium(outsideMedium) {}

    std::string ToString() const {
        return StringPrintf(
            "[ ShapeSeneEntity name: %s parameters: %s loc: %s "
            "renderFromObject: %s reverseOrientation: %s materialIndex: %d "
            "materialName: %s insideMedium: %s outsideMedium: %s]",
            name, parameters, loc, renderFromObject, reverseOrientation, materialIndex,
            materialName, insideMedium, outsideMedium);
    }

    const Transform *identity = nullptr;
    bool reverseOrientation = false;
    int materialIndex;  // one of these two...  std::variant?
    std::string materialName;
    int lightIndex = -1;
    std::string insideMedium, outsideMedium;
};

struct InstanceDefinitionSceneEntity {
    InstanceDefinitionSceneEntity() = default;
    InstanceDefinitionSceneEntity(const std::string &name, FileLoc loc)
        : name(name), loc(loc) {}

    std::string ToString() const {
        return StringPrintf("[ InstanceDefinitionSceneEntity name: %s loc: %s "
                            " shapes: %s animatedShapes: %s ]",
                            name, loc, shapes, animatedShapes);
    }

    std::string name;
    FileLoc loc;
    std::vector<ShapeSceneEntity> shapes;
    std::vector<AnimatedShapeSceneEntity> animatedShapes;
};

struct TextureSceneEntity : public TransformedSceneEntity {
    TextureSceneEntity() = default;
    TextureSceneEntity(const std::string &texName, ParameterDictionary parameters,
                       FileLoc loc, const AnimatedTransform &renderFromObject)
        : TransformedSceneEntity("", std::move(parameters), loc, renderFromObject),
          texName(texName) {}

    std::string ToString() const {
        return StringPrintf("[ TextureSeneEntity name: %s parameters: %s loc: %s "
                            "renderFromObject: %s texName: %s ]",
                            name, parameters, loc, renderFromObject, texName);
    }

    std::string texName;
};

struct LightSceneEntity : public TransformedSceneEntity {
    LightSceneEntity() = default;
    LightSceneEntity(const std::string &name, ParameterDictionary parameters, FileLoc loc,
                     const AnimatedTransform &renderFromLight, const std::string &medium)
        : TransformedSceneEntity(name, parameters, loc, renderFromLight),
          medium(medium) {}

    std::string ToString() const {
        return StringPrintf("[ LightSeneEntity name: %s parameters: %s loc: %s "
                            "renderFromObject: %s medium: %s ]",
                            name, parameters, loc, renderFromObject, medium);
    }

    std::string medium;
};

struct InstanceSceneEntity : public SceneEntity {
    InstanceSceneEntity() = default;
    InstanceSceneEntity(const std::string &name, FileLoc loc,
                        const AnimatedTransform &renderFromInstanceAnim)
        : SceneEntity(name, {}, loc),
          renderFromInstanceAnim(new AnimatedTransform(renderFromInstanceAnim)) {}
    InstanceSceneEntity(const std::string &name, FileLoc loc,
                        const Transform *renderFromInstance)
        : SceneEntity(name, {}, loc), renderFromInstance(renderFromInstance) {}

    std::string ToString() const {
        return StringPrintf(
            "[ InstanceSeneEntity name: %s loc: %s "
            "renderFromInstanceAnim: %s renderFromInstance: %s ]",
            name, loc,
            renderFromInstanceAnim ? renderFromInstanceAnim->ToString()
                                   : std::string("nullptr"),
            renderFromInstance ? renderFromInstance->ToString() : std::string("nullptr"));
    }

    AnimatedTransform *renderFromInstanceAnim;
    const Transform *renderFromInstance;
};

// TransformHash Definition
struct TransformHash {
    size_t operator()(const Transform *t) const { return t->Hash(); }
};

// TransformCache Definition
class TransformCache {
  public:
    // TransformCache Public Methods
    TransformCache()
        : bufferResource(Options->useGPU ? gpuMemoryAllocator.resource()
                                         : Allocator().resource()),
          alloc(&bufferResource) {}
    ~TransformCache();

    const Transform *Lookup(const Transform &t);

  private:
    // TransformCache Private Members
    pstd::pmr::monotonic_buffer_resource bufferResource;
    Allocator alloc;
    std::unordered_set<Transform *, TransformHash> hashTable;
};

// MaxTransforms Definition
constexpr int MaxTransforms = 2;

// TransformSet Definition
struct TransformSet {
    // TransformSet Public Methods
    Transform &operator[](int i) {
        CHECK_GE(i, 0);
        CHECK_LT(i, MaxTransforms);
        return t[i];
    }
    const Transform &operator[](int i) const {
        CHECK_GE(i, 0);
        CHECK_LT(i, MaxTransforms);
        return t[i];
    }
    friend TransformSet Inverse(const TransformSet &ts) {
        TransformSet tInv;
        for (int i = 0; i < MaxTransforms; ++i)
            tInv.t[i] = Inverse(ts.t[i]);
        return tInv;
    }
    bool IsAnimated() const {
        for (int i = 0; i < MaxTransforms - 1; ++i)
            if (t[i] != t[i + 1])
                return true;
        return false;
    }

  private:
    Transform t[MaxTransforms];
};

// ParsedScene Definition
class ParsedScene : public SceneRepresentation {
  public:
    // ParsedScene Public Methods
    ParsedScene();
    void Option(const std::string &name, const std::string &value, FileLoc loc);
    void Identity(FileLoc loc);
    void Translate(Float dx, Float dy, Float dz, FileLoc loc);
    void Rotate(Float angle, Float ax, Float ay, Float az, FileLoc loc);
    void Scale(Float sx, Float sy, Float sz, FileLoc loc);
    void LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz, Float ux,
                Float uy, Float uz, FileLoc loc);
    void ConcatTransform(Float transform[16], FileLoc loc);
    void Transform(Float transform[16], FileLoc loc);
    void CoordinateSystem(const std::string &, FileLoc loc);
    void CoordSysTransform(const std::string &, FileLoc loc);
    void ActiveTransformAll(FileLoc loc);
    void ActiveTransformEndTime(FileLoc loc);
    void ActiveTransformStartTime(FileLoc loc);
    void TransformTimes(Float start, Float end, FileLoc loc);
    void ColorSpace(const std::string &n, FileLoc loc);
    void PixelFilter(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Film(const std::string &type, ParsedParameterVector params, FileLoc loc);
    void Sampler(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Accelerator(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Integrator(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Camera(const std::string &, ParsedParameterVector params, FileLoc loc);
    void MakeNamedMedium(const std::string &name, ParsedParameterVector params,
                         FileLoc loc);
    void MediumInterface(const std::string &insideName, const std::string &outsideName,
                         FileLoc loc);
    void WorldBegin(FileLoc loc);
    void AttributeBegin(FileLoc loc);
    void AttributeEnd(FileLoc loc);
    void Attribute(const std::string &target, ParsedParameterVector params, FileLoc loc);
    void Texture(const std::string &name, const std::string &type,
                 const std::string &texname, ParsedParameterVector params, FileLoc loc);
    void Material(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void MakeNamedMaterial(const std::string &name, ParsedParameterVector params,
                           FileLoc loc);
    void NamedMaterial(const std::string &name, FileLoc loc);
    void LightSource(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void AreaLightSource(const std::string &name, ParsedParameterVector params,
                         FileLoc loc);
    void Shape(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void ReverseOrientation(FileLoc loc);
    void ObjectBegin(const std::string &name, FileLoc loc);
    void ObjectEnd(FileLoc loc);
    void ObjectInstance(const std::string &name, FileLoc loc);

    void EndOfFiles();

    ParsedScene *CopyForImport();
    void MergeImported(ParsedScene *);

    std::string ToString() const;

    NamedTextures CreateTextures(Allocator alloc, bool gpu) const;

    void CreateMaterials(/*const*/ NamedTextures &sceneTextures, Allocator alloc,
                         std::map<std::string, pbrt::Material> *namedMaterials,
                         std::vector<pbrt::Material> *materials) const;

    std::map<std::string, Medium> CreateMedia(Allocator alloc) const;

    // ParsedScene Public Members
    SceneEntity film, sampler, integrator, filter, accelerator;
    CameraSceneEntity camera;
    std::vector<std::pair<std::string, SceneEntity>> namedMaterials;
    std::set<std::string> namedMaterialNames;
    std::vector<SceneEntity> materials;
    std::map<std::string, TransformedSceneEntity> media;
    std::vector<std::pair<std::string, TextureSceneEntity>> floatTextures;
    std::vector<std::pair<std::string, TextureSceneEntity>> spectrumTextures;
    std::set<std::string> floatTextureNames, spectrumTextureNames;
    std::vector<LightSceneEntity> lights;
    std::vector<SceneEntity> areaLights;
    std::vector<ShapeSceneEntity> shapes;
    std::vector<AnimatedShapeSceneEntity> animatedShapes;
    std::vector<InstanceSceneEntity> instances;
    std::map<std::string, InstanceDefinitionSceneEntity> instanceDefinitions;

  private:
    // ParsedScene::GraphicsState Definition
    struct GraphicsState {
        // GraphicsState Public Methods
        GraphicsState();

        // GraphicsState Public Members
        std::string currentInsideMedium, currentOutsideMedium;

        int currentMaterialIndex = 0;
        std::string currentMaterialName;

        std::string areaLightName;
        ParameterDictionary areaLightParams;
        FileLoc areaLightLoc;

        ParsedParameterVector shapeAttributes;
        ParsedParameterVector lightAttributes;
        ParsedParameterVector materialAttributes;
        ParsedParameterVector mediumAttributes;
        ParsedParameterVector textureAttributes;
        bool reverseOrientation = false;
        const RGBColorSpace *colorSpace = RGBColorSpace::sRGB;
        TransformSet ctm;
        uint32_t activeTransformBits = AllTransformsBits;
        Float transformStartTime = 0, transformEndTime = 1;
    };

    friend void parse(SceneRepresentation *scene, std::unique_ptr<Tokenizer> t);
    // ParsedScene Private Methods
    class Transform RenderFromObject(int index) const {
        return pbrt::Transform((renderFromWorld * graphicsState.ctm[index]).GetMatrix());
    }

    AnimatedTransform RenderFromObject() const {
        return {RenderFromObject(0), graphicsState.transformStartTime,
                RenderFromObject(1), graphicsState.transformEndTime};
    }

    bool CTMIsAnimated() const { return graphicsState.ctm.IsAnimated(); }

    // ParsedScene Private Members
    GraphicsState graphicsState;
    enum class BlockState { OptionsBlock, WorldBlock };
    BlockState currentBlock = BlockState::OptionsBlock;
    static constexpr int StartTransformBits = 1 << 0;
    static constexpr int EndTransformBits = 1 << 1;
    static constexpr int AllTransformsBits = (1 << MaxTransforms) - 1;
    std::map<std::string, TransformSet> namedCoordinateSystems;
    class Transform renderFromWorld;
    TransformCache transformCache;
    std::vector<GraphicsState> pushedGraphicsStates;
    std::vector<std::pair<char, FileLoc>> pushStack;  // 'a': attribute, 'o': object
    InstanceDefinitionSceneEntity *currentInstance = nullptr;
};

class FormattingScene : public SceneRepresentation {
  public:
    FormattingScene(bool toPly, bool upgrade) : toPly(toPly), upgrade(upgrade) {}
    ~FormattingScene();

    void Option(const std::string &name, const std::string &value, FileLoc loc);
    void Identity(FileLoc loc);
    void Translate(Float dx, Float dy, Float dz, FileLoc loc);
    void Rotate(Float angle, Float ax, Float ay, Float az, FileLoc loc);
    void Scale(Float sx, Float sy, Float sz, FileLoc loc);
    void LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz, Float ux,
                Float uy, Float uz, FileLoc loc);
    void ConcatTransform(Float transform[16], FileLoc loc);
    void Transform(Float transform[16], FileLoc loc);
    void CoordinateSystem(const std::string &, FileLoc loc);
    void CoordSysTransform(const std::string &, FileLoc loc);
    void ActiveTransformAll(FileLoc loc);
    void ActiveTransformEndTime(FileLoc loc);
    void ActiveTransformStartTime(FileLoc loc);
    void TransformTimes(Float start, Float end, FileLoc loc);
    void TransformBegin(FileLoc loc);
    void TransformEnd(FileLoc loc);
    void ColorSpace(const std::string &n, FileLoc loc);
    void PixelFilter(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Film(const std::string &type, ParsedParameterVector params, FileLoc loc);
    void Sampler(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Accelerator(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Integrator(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Camera(const std::string &, ParsedParameterVector params, FileLoc loc);
    void MakeNamedMedium(const std::string &name, ParsedParameterVector params,
                         FileLoc loc);
    void MediumInterface(const std::string &insideName, const std::string &outsideName,
                         FileLoc loc);
    void WorldBegin(FileLoc loc);
    void AttributeBegin(FileLoc loc);
    void AttributeEnd(FileLoc loc);
    void Attribute(const std::string &target, ParsedParameterVector params, FileLoc loc);
    void Texture(const std::string &name, const std::string &type,
                 const std::string &texname, ParsedParameterVector params, FileLoc loc);
    void Material(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void MakeNamedMaterial(const std::string &name, ParsedParameterVector params,
                           FileLoc loc);
    void NamedMaterial(const std::string &name, FileLoc loc);
    void LightSource(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void AreaLightSource(const std::string &name, ParsedParameterVector params,
                         FileLoc loc);
    void Shape(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void ReverseOrientation(FileLoc loc);
    void ObjectBegin(const std::string &name, FileLoc loc);
    void ObjectEnd(FileLoc loc);
    void ObjectInstance(const std::string &name, FileLoc loc);

    void EndOfFiles();

    std::string indent(int extra = 0) const {
        return std::string(catIndentCount + 4 * extra, ' ');
    }

  private:
    std::string upgradeMaterialIndex(const std::string &name, ParameterDictionary *dict,
                                     FileLoc loc) const;
    std::string upgradeMaterial(std::string *name, ParameterDictionary *dict,
                                FileLoc loc) const;

    int catIndentCount = 0;
    bool toPly, upgrade;
    std::map<std::string, std::string> definedTextures;
    std::map<std::string, std::string> definedNamedMaterials;
    std::map<std::string, ParameterDictionary> namedMaterialDictionaries;
    std::map<std::string, std::string> definedObjectInstances;
};

}  // namespace pbrt

#endif  // PBRT_PARSEDSCENE_H
