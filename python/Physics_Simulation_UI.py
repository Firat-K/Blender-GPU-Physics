import bpy
import numpy as np
import time
import itertools

class PhysicsSettings(bpy.types.PropertyGroup):
    phys_freq: bpy.props.IntProperty(name="Frequency", default=60)
    phys_iter: bpy.props.IntProperty(name="Total Iterations", default=100)
    phys_frames: bpy.props.IntProperty(name="Frames", default=100)
    frame_start: bpy.props.IntProperty(name="Frame Start", default=1)
    frame_end: bpy.props.IntProperty(name="Frame End", default=250)
    gravity: bpy.props.FloatVectorProperty(name="Gravity", size=3, default=(0.0, 0.0, -9.81))
    floor_friction: bpy.props.FloatProperty(name="Floor friction", default=0.5)
    solver_path: bpy.props.StringProperty(name="Solver Path", default="")


class SimplePanel(bpy.types.Panel):
    """Creates a Simple Panel in the Scene properties window"""
    bl_label = "Physics Settings"
    bl_idname = "SCENE_PT_physics_settings"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Custom properties
        physics_settings = scene.physics_settings

        # Integer fields
        layout.prop(physics_settings, "phys_freq")
        layout.prop(physics_settings, "phys_iter")
        layout.prop(physics_settings, "phys_frames")
        layout.prop(physics_settings, "frame_start")
        layout.prop(physics_settings, "frame_end")

        # Float fields
        layout.prop(physics_settings, "gravity")
        layout.prop(physics_settings, "floor_friction")

        # String field
        layout.prop(physics_settings, "solver_path")


class SolvePhysicsOperator(bpy.types.Operator):
    """Operator to execute the Physics_Environment script"""
    bl_idname = "scene.solve_physics"
    bl_label = "Solve"

    def execute(self, context):
        # Execute the "Physics_Environment" script
        exec(bpy.data.texts['Physics_Environment'].as_string())
        return {'FINISHED'}


def register():
    bpy.utils.register_class(PhysicsSettings)
    bpy.utils.register_class(SimplePanel)
    bpy.utils.register_class(SolvePhysicsOperator)
    bpy.types.Scene.physics_settings = bpy.props.PointerProperty(type=PhysicsSettings)


def unregister():
    bpy.utils.unregister_class(PhysicsSettings)
    bpy.utils.unregister_class(SimplePanel)
    bpy.utils.unregister_class(SolvePhysicsOperator)
    del bpy.types.Scene.physics_settings


if __name__ == "__main__":
    register()
