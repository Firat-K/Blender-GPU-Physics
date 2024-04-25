import bpy
import numpy as np

def update_progress_bar(progress):
    progress_length = 40
    filled_length = int(progress * progress_length)
    bar = '█' * filled_length + '-' * (progress_length - filled_length)
    print(f'\rBinding Progress: |{bar}| {progress * 100:.2f}% Complete', end='', flush=True)

def sub_vectors(a, b):
    return [a[i] - b[i] for i in range(3)]

def tet_volume(vec1, vec2, vec3):
    return np.dot(vec1, np.cross(vec2, vec3)) / 6.0

def get_barycentric_coordinates(p0, p1, p2, p3, t):
    vec01 = np.array(sub_vectors(p1, p0))
    vec02 = np.array(sub_vectors(p2, p0))
    vec03 = np.array(sub_vectors(p3, p0))
    vec0t = np.array(sub_vectors(t, p0))

    vol = tet_volume(vec01, vec02, vec03)
    vol1 = tet_volume(vec01, vec02, vec0t)
    vol2 = tet_volume(vec02, vec03, vec0t)
    vol3 = tet_volume(vec03, vec01, vec0t)

    bary_coords = [
        vol1 / vol,
        vol2 / vol,
        vol3 / vol
    ]
    return bary_coords

def bind_tet_mesh(self, context):
    
    target_obj = bpy.data.objects.get(context.object.target_tet_mesh)
    
    bind_vert_list = []
    bind_weight_list = []
    bind_coord_list = [co for vertex in target_obj.data.vertices for co in vertex.co]
    
    if target_obj is None:
        self.report({'ERROR'}, "Target Tet Mesh not found")
        return
    
    for i, vertex in enumerate(context.object.data.vertices):
        t = [vertex.co[0], vertex.co[1], vertex.co[2]]
        is_hit = False
        for tet in target_obj.data.polygons:
            tet_v = [tet.vertices[0], tet.vertices[1], tet.vertices[2], tet.vertices[3]]
            p1 = [target_obj.data.vertices[tet_v[0]].co[0], target_obj.data.vertices[tet_v[0]].co[1], target_obj.data.vertices[tet_v[0]].co[2]]
            p2 = [target_obj.data.vertices[tet_v[1]].co[0], target_obj.data.vertices[tet_v[1]].co[1], target_obj.data.vertices[tet_v[1]].co[2]]
            p3 = [target_obj.data.vertices[tet_v[2]].co[0], target_obj.data.vertices[tet_v[2]].co[1], target_obj.data.vertices[tet_v[2]].co[2]]
            p4 = [target_obj.data.vertices[tet_v[3]].co[0], target_obj.data.vertices[tet_v[3]].co[1], target_obj.data.vertices[tet_v[3]].co[2]]
            
            bar_coords = get_barycentric_coordinates(p1, p2, p3, p4, t)
            if max(bar_coords) <= 1.0 and min(bar_coords) >= 0.0 and sum(bar_coords) <= 1.0 and sum(bar_coords) >= 0.0:
                is_hit = True
                
                bind_vert_list.append(tet_v[0])
                bind_vert_list.append(tet_v[1])
                bind_vert_list.append(tet_v[2])
                bind_vert_list.append(tet_v[3])
                
                bind_weight_list.append(bar_coords[0])
                bind_weight_list.append(bar_coords[1])
                bind_weight_list.append(bar_coords[2])
                bind_weight_list.append(1 - sum(bar_coords))
                break
        
        if is_hit == False:
            bind_vert_list.append(-1)
            bind_vert_list.append(-1)
            bind_vert_list.append(-1)
            bind_vert_list.append(-1)
            bind_weight_list.append(0.0)
            bind_weight_list.append(0.0)
            bind_weight_list.append(0.0)
            bind_weight_list.append(0.0)
        progress = (i + 1) / len(context.object.data.vertices)
        update_progress_bar(progress)
    
    context.object['bind_verts'] = bind_vert_list
    context.object['bind_weights'] = bind_weight_list
    context.object['bind_coords'] = bind_coord_list
    context.object['is_bound'] = True
    
def update_bind_deformations(self, context):
    target_obj = bpy.data.objects.get(context.object.target_tet_mesh)
    b_verts = context.object.get('bind_verts',[]).to_list()
    b_weights = context.object.get('bind_weights', []).to_list()
    b_coords = context.object.get('bind_coords', []).to_list()
    
    t_coords = [co for vertex in target_obj.data.vertices for co in vertex.co]
    t_disp = [x - y for x, y in zip(t_coords, b_coords)]
    
    for i, vertex in enumerate(context.object.data.vertices):
        if sum(b_coords)  == 0.0:
            break
        
        bound_verts = [b_verts[i * 4], b_verts[i * 4 + 1], b_verts[i * 4 + 2], b_verts[i * 4 + 3]]
        bound_weights = [b_weights[i * 4], b_weights[i * 4 + 1], b_weights[i * 4 + 2], b_weights[i * 4 + 3]]
        
        d1 = [t_disp[bound_verts[0] * 3], t_disp[bound_verts[0] * 3 + 1], t_disp[bound_verts[0] * 3 + 2]]
        d2 = [t_disp[bound_verts[1] * 3], t_disp[bound_verts[1] * 3 + 1], t_disp[bound_verts[1] * 3 + 2]]
        d3 = [t_disp[bound_verts[2] * 3], t_disp[bound_verts[2] * 3 + 1], t_disp[bound_verts[2] * 3 + 2]]
        d4 = [t_disp[bound_verts[3] * 3], t_disp[bound_verts[3] * 3 + 1], t_disp[bound_verts[3] * 3 + 2]]
        
        vertex.co[0] += (d1[0] * bound_weights[0]) + (d2[0] * bound_weights[1]) + (d3[0] * bound_weights[2]) +(d4[0] * bound_weights[3])
        vertex.co[1] += (d1[1] * bound_weights[0]) + (d2[1] * bound_weights[1]) + (d3[1] * bound_weights[2]) +(d4[1] * bound_weights[3])
        vertex.co[2] += (d1[2] * bound_weights[0]) + (d2[2] * bound_weights[1]) + (d3[2] * bound_weights[2]) +(d4[2] * bound_weights[3])
    
    context.object['bind_coords'] = [co for vertex in target_obj.data.vertices for co in vertex.co]

# Define the panel class
class OBJECT_PT_TetMeshBindingPanel(bpy.types.Panel):
    bl_label = "Tet Mesh Binding"
    bl_idname = "OBJECT_PT_tet_mesh_binding"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout

        # Add an object field for selecting the target tet mesh
        layout.prop_search(context.object, "target_tet_mesh", bpy.data, "objects", text="Target Tet Mesh")

        # Add a button to execute the bind function
        layout.operator("object.bind_operator")

        # Add a button to execute the update_bind_deformations function
        layout.operator("object.update_deformations_operator")

# Define the operator class
class OBJECT_OT_BindOperator(bpy.types.Operator):
    bl_label = "Bind"
    bl_idname = "object.bind_operator"

    def execute(self, context):
        bind_tet_mesh(self, context)
        return {'FINISHED'}

# Define the operator class to update deformations
class OBJECT_OT_UpdateDeformationsOperator(bpy.types.Operator):
    bl_label = "Update Deformations"
    bl_idname = "object.update_deformations_operator"

    def execute(self, context):
        update_bind_deformations(self, context)
        return {'FINISHED'}

# Register classes
def register():
    bpy.utils.register_class(OBJECT_PT_TetMeshBindingPanel)
    bpy.utils.register_class(OBJECT_OT_BindOperator)
    bpy.utils.register_class(OBJECT_OT_UpdateDeformationsOperator)
    bpy.types.Object.target_tet_mesh = bpy.props.StringProperty()

def unregister():
    bpy.utils.unregister_class(OBJECT_PT_TetMeshBindingPanel)
    bpy.utils.unregister_class(OBJECT_OT_BindOperator)
    bpy.utils.unregister_class(OBJECT_OT_UpdateDeformationsOperator)
    del bpy.types.Object.target_tet_mesh

# Run the script
if __name__ == "__main__":
    register()
