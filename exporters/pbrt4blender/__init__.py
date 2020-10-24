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


bl_info = {
    "name"       : "Pbrt4Blender",
    "description": "pbrt v4 integration for Blender",
    "author"     : "Pedro A. 'povmaniac'",
    "version"    : (0, 0, 0, 1),
    "blender"    : (2, 90, 1),
    "location"   : "Info Header > Engine dropdown menu",
    "wiki_url"   : "",
    "tracker_url": "",
    "category"   : "Render",
}

import sys
import os

BIN_PATH = os.path.join(__path__[0], 'bin')
sys.path.insert(0, BIN_PATH)
os.environ['PATH'] = BIN_PATH + ';' + os.environ['PATH']

#--------------------------
# import modules
#--------------------------
if "bpy" in locals():
    import importlib
    if "ui" in locals:
        importlib.reload(ui)
    if "io" in locals:
        importlib.reload(io)


import sys
import os 
import bpy
import bgl
import threading
from bpy_extras import view3d_utils
import math
import gpu

from . import utils


class Pbrt4RenderEngine(bpy.types.RenderEngine):
    # These three members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = "PBRT4"
    bl_label = "Pbrt4Blender"
    bl_use_preview = True


    # Init is called whenever a new render engine instance is created.
    # Multiple instances may exist at the same time, for example for 
    # a viewport and final render.
    def __init__(self):
        print('INIT ENGINE..')
        

    # When the render engine instance is destroy, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    def __del__(self):
        print('DEL ENGINE..')


    def beautyRender(self, folder):
        """Call pbrt binary with some hard-coded parameters"""
        os.chdir(folder)
        # call a bit modified killeroos scene
        os.system('pbrt ../kill/kill.pbrt')

    # This is the method called by Blender for both final renders (F12) and
    # small preview for materials, world and lights.
    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)
        

        if not self.is_preview:
            #
            result = self.begin_result(0, 0, 700, 700)
            #
            self.beautyRender(BIN_PATH)
            #
            lay = result.layers[0]
            lay.load_from_file(BIN_PATH +"/killeroo-simple.exr")         
            #
            self.end_result(result) 
        
        else:
            # Fill the render result with a flat color. The framebuffer is
            # defined as a list of pixels, each pixel itself being a list of
            # R,G,B,A values.
            color = [0.2, 0.1, 0.1, 1.0]

            pixel_count = self.size_x * self.size_y
            rect = [color] * pixel_count

            # Here we write the pixel values to the RenderResult
            result = self.begin_result(0, 0, self.size_x, self.size_y)
            layer = result.layers[0].passes["Combined"]
            layer.rect = rect
            self.end_result(result)
        

    # For viewport renders, this method gets called once at the start and
    # whenever the scene or 3D viewport changes. This method is where data
    # should be read from Blender in the same thread. Typically a render
    # thread will be started to do the work while keeping Blender responsive.
    def _view_update(self, context, depsgraph):
        print('UPDATING..')
        
        
    # For viewport renders, this method is called whenever Blender redraws
    # the 3D viewport. The renderer is expected to quickly draw the render
    # with OpenGL, and not perform other expensive work.
    # Blender will draw overlays for selection and editing on top of the
    # rendered image automatically.
    def _view_draw(self, context, depsgraph):
        print("DRAWING..")



def register():
    #
    from . import ui, io
    ui.register()
    io.register()

    # Register the RenderEngine
    bpy.utils.register_class(Pbrt4RenderEngine)


def unregister():
    #
    ui.unregister()
    io.unregister()

    bpy.utils.unregister_class(Pbrt4RenderEngine)


if __name__ == "__main__":
    register()

