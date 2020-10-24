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

# <pep8 compliant>

import bpy, bl_ui
from bpy.types import Panel, Menu
classes=[]

class RenderButtonsPanel():
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'PBRT4'}
    
    @classmethod
    def poll(cls, context):
        scene = context.scene
        return scene and (scene.render.engine == "PBRT4")
   
class PBRT4_PT_Render_Panel(RenderButtonsPanel, Panel):
    #
    bl_label = "Render"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'PBRT4'}
    
    @classmethod
    def poll(cls, context):
        scene = context.scene
        return context.scene and (context.scene.render.engine in cls.COMPAT_ENGINES)
    
    def draw(self, context):
        layout = self.layout

        row = layout.row(align=True)
        row.operator("render.render", text="Render", icon='RENDER_STILL')
        row.operator("render.render", text="Animation", icon='RENDER_ANIMATION').animation = True
        #
        rd = context.scene.render
        row = layout.row()
        view = bpy.context.preferences.view
        row.prop(view, "render_display_type", text='Display')
        row.prop(rd, "use_lock_interface", text="", emboss=False, icon='DECORATE_UNLOCKED')      

classes.append(PBRT4_PT_Render_Panel)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    
def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    

#
if __name__== "__main__":
    register()