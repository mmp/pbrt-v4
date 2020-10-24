#-----------------------------------------------------------------------------#
#
#    This file is part of BountyPbrt4, the Pbrt v4 integration for Blender
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

import bpy
from . import ui_render


# RenderEngines also need to tell UI Panels that they are compatible with.
# We recommend to enable all panels marked as BLENDER_RENDER, and then
# exclude any panels that are replaced by custom panels registered by the
# render engine, or that are not supported.

exclude_panels = [
        'VIEWLAYER_PT_eevee_layer_passes',
        'VIEWLAYER_PT_eevee_layer_passes_data',
        'VIEWLAYER_PT_eevee_layer_passes_light',
        'VIEWLAYER_PT_eevee_layer_passes_effects',
        'RENDER_PT_simplify',
        'RENDER_PT_simplify_viewport',
        'RENDER_PT_simplify_render',
        'RENDER_PT_simplify_greasepencil',
        'RENDER_PT_color_management',
        'RENDER_PT_color_management_curves',
        'RENDER_PT_stamp',
        'RENDER_PT_stamp_note',
        'RENDER_PT_gpencil',
        'EEVEE_MATERIAL_PT_viewport_settings',
        'EEVEE_MATERIAL_PT_settings',
        'DATA_PT_lens',
        'DATA_PT_camera',
        'DATA_PT_camera_display',
        'DATA_PT_camera_display_composition_guides',
        'DATA_PT_camera_display_passepartout',
        'DATA_PT_preview',
        'DATA_PT_light',
        'DATA_PT_area',
        'WORLD_PT_context_world',
        'WORLD_PT_viewport_display',
        'TEXTURE_PT_context',
        'TEXTURE_PT_image',
        'TEXTURE_PT_image_settings',
        'TEXTURE_PT_image_alpha',
        'TEXTURE_PT_image_mapping',
        'TEXTURE_PT_image_mapping_crop',
        'TEXTURE_PT_image_sampling',
        'TEXTURE_PT_colors',
        #'TEXTURE_PT_colors_ramp',
        
    ]
def get_panels():  

    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
            if panel.__name__ not in exclude_panels and not 'custom' in panel.__name__:
                panels.append(panel)

    return panels

def register():
    #------------------------------
    ui_render.register()
    #
    for panel in get_panels():
        panel.COMPAT_ENGINES.add('PBRT4')


def unregister():
    #
    ui_render.unregister()
    #
    for panel in get_panels():
        panel.COMPAT_ENGINES.remove('PBRT4')

#EOF 95