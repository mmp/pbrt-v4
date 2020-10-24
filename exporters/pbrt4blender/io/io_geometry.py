#-----------------------------------------------------------------------------#
#
#    This file is part of Pbrt4Blender, the Pbrt-v4 integration for Blender
#
#    Copyright 2020, Pedro A. "povmaniac"
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#-----------------------------------------------------------------------------#
 
# test
scene_data = '''
LookAt 400 20 30
    0 63 -110
    0 0 1
Rotate -5 0 0 1
Camera "perspective"
    "float fov" [ 39 ]
# zoom in by feet
# "integer xresolution" [1500] "integer yresolution" [1500]
#	"float cropwindow" [ .34 .49  .67 .8 ]
Film "rgb"
    "string filename" [ "killeroo-simple.exr" ]
    "integer yresolution" [ 700 ]
    "integer xresolution" [ 700 ]
Sampler "halton"
    "integer pixelsamples" [ 64 ]


WorldBegin


AttributeBegin
    Material "diffuse"
        "rgb reflectance" [ 0 0 0 ]
    Translate 150 0 20
    Translate 0 120 0
    AreaLightSource "diffuse"
        "rgb L" [ 2000 2000 2000 ]
    Shape "sphere"
        "float radius" [ 3 ]
AttributeEnd

AttributeBegin
    Material "diffuse"
        "rgb reflectance" [ 0.5 0.5 0.8 ]
    Translate 0 0 -140
    Shape "trianglemesh"
        "point2 uv" [ 0 0 5 0 5 5 0 5 
            ]
        "integer indices" [ 0 1 2 2 3 0 ]
        "point3 P" [ -1000 -1000 0 1000 -1000 0 1000 1000 0 -1000 1000 0 ]
    Shape "trianglemesh"
        "point2 uv" [ 0 0 5 0 5 5 0 5 
            ]
        "integer indices" [ 0 1 2 2 3 0 ]
        "point3 P" [ -400 -1000 -1000 -400 1000 -1000 -400 1000 1000 -400 -1000 1000 ]
AttributeEnd

AttributeBegin
    Scale 0.5 0.5 0.5
    Rotate -60 0 0 1
    Material "coateddiffuse"
        "float roughness" [ 0.025 ]
        "rgb reflectance" [ 0.4 0.2 0.2 ]
    Translate 100 200 -140
    Include "geometry/killeroo.pbrt"
    Material "coateddiffuse"
        "float roughness" [ 0.15 ]
        "rgb reflectance" [ 0.4 0.5 0.4 ]
    Translate -200 0 0
    Include "geometry/killeroo.pbrt"
AttributeEnd
'''

# test end