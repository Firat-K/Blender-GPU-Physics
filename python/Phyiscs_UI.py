import bpy

# Define the panel class
class OBJECT_PT_PhysObjPanel(bpy.types.Panel):
    bl_label = "Physics Mesh Properties"
    bl_idname = "OBJECT_PT_PhysObjPanel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"

    # Define boolean properties
    bpy.types.Object.show_compliance_properties = bpy.props.BoolProperty(name="Show Compliance Properties")
    bpy.types.Object.show_damping_properties = bpy.props.BoolProperty(name="Show Damping Properties")
    bpy.types.Object.show_vertex_properties = bpy.props.BoolProperty(name="Show Vertex Collider Properties")

    def draw(self, context):
        layout = self.layout
        obj = context.object

        # Dropdown menu
        layout.label(text="Physics Type:")
        layout.prop(obj, "phys_type", text="")

        # Toggle for Compliance properties
        layout.prop(obj, "show_compliance_properties", text="Compliance Properties", icon='TRIA_DOWN' if obj.show_compliance_properties else 'TRIA_RIGHT', emboss=False)

        if obj.show_compliance_properties:
            self.draw_compliance_properties(layout, obj)

        # Toggle for Damping properties
        layout.prop(obj, "show_damping_properties", text="Damping Properties", icon='TRIA_DOWN' if obj.show_damping_properties else 'TRIA_RIGHT', emboss=False)

        if obj.show_damping_properties:
            self.draw_damping_properties(layout, obj)

        # Toggle for Vertex Collider properties
        layout.prop(obj, "show_vertex_properties", text="Collision Properties", icon='TRIA_DOWN' if obj.show_vertex_properties else 'TRIA_RIGHT', emboss=False)

        if obj.show_vertex_properties:
            self.draw_vertex_properties(layout, obj)

    def draw_compliance_properties(self, layout, obj):
        layout.label(text="Edge Tension Compliance:")
        layout.prop(obj, "edge_tens_comp", text="")

        layout.label(text="Edge Compression Compliance:")
        layout.prop(obj, "edge_comp_comp", text="")

        layout.label(text="Volume Compliance:")
        layout.prop(obj, "vol_comp", text="")

        layout.label(text="Bending Compliance:")
        layout.prop(obj, "bend_comp", text="")

    def draw_damping_properties(self, layout, obj):
        layout.label(text="Edge Damping:")
        layout.prop(obj, "edge_damp", text="")

        layout.label(text="Volume Damping:")
        layout.prop(obj, "vol_damp", text="")

        layout.label(text="Bending Damping:")
        layout.prop(obj, "bend_damp", text="")

    def draw_vertex_properties(self, layout, obj):
        layout.label(text="Vertex Collider Radius Multiplier:")
        layout.prop(obj, "vert_rad_mult", text="")

# Define the menu items
def update_menu(self, context):
    pass

# Properties for the float fields
bpy.types.Object.phys_type = bpy.props.EnumProperty(
    items=[
        ("NON", "None", ""),
        ("TET", "Tetrahedral", ""),
        ("CLOTH", "Cloth", ""),
        ("MAN", "Manifold", ""),
        ("STAT", "Static", "")
    ],
    update=update_menu
)

bpy.types.Object.edge_tens_comp = bpy.props.FloatProperty(name="Edge Tension Compliance")
bpy.types.Object.edge_comp_comp = bpy.props.FloatProperty(name="Edge Compression Compliance")
bpy.types.Object.vol_comp = bpy.props.FloatProperty(name="Volume Compliance")
bpy.types.Object.bend_comp = bpy.props.FloatProperty(name="Bending Compliance")
bpy.types.Object.edge_damp = bpy.props.FloatProperty(name="Edge Damping")
bpy.types.Object.vol_damp = bpy.props.FloatProperty(name="Volume Damping")
bpy.types.Object.bend_damp = bpy.props.FloatProperty(name="Bending Damping")
bpy.types.Object.vert_rad_mult = bpy.props.FloatProperty(name="Vertex Collider Radius Multiplier")

# Register
def register():
    bpy.utils.register_class(OBJECT_PT_PhysObjPanel)

def unregister():
    bpy.utils.unregister_class(OBJECT_PT_PhysObjPanel)

if __name__ == "__main__":
    register()
