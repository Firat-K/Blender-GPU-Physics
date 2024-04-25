import bpy
import numpy as np
import time
import itertools
import ctypes
import bmesh
import os

def update_progress_bar(progress):
    progress_length = 40
    filled_length = int(progress * progress_length)
    bar = '█' * filled_length + '-' * (progress_length - filled_length)
    print(f'\rSimulation Progress: |{bar}| {progress * 100:.2f}% Complete', end='', flush=False)
    print()

class PhysicsSettings(bpy.types.PropertyGroup):
    phys_freq: bpy.props.IntProperty(name="Frequency", default=60)
    phys_iter: bpy.props.IntProperty(name="Total Iterations", default=100)
    phys_frames: bpy.props.IntProperty(name="Frames", default=100)
    frame_start: bpy.props.IntProperty(name="Frame Start", default=1)
    frame_end: bpy.props.IntProperty(name="Frame End", default=250)
    gravity: bpy.props.FloatVectorProperty(name="Gravity", size=3, default=(0.0, 0.0, -9.81))
    floor_friction: bpy.props.FloatProperty(name="Floor friction", default=0.5)
    solver_path: bpy.props.StringProperty(name="Solver Path", default="")
    
    # Add a CollectionProperty for the physics collection
    physics_collection: bpy.props.PointerProperty(
        type=bpy.types.Collection,
        name="Physics Collection",
    )


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
        
        # Collection field
        layout.prop(physics_settings, "physics_collection")

        # Solve button
        layout.operator("scene.solve_physics", text="Solve")
        
class SolvePhysicsOperator(bpy.types.Operator):
    """Operator to execute the Physics_Environment script"""
    bl_idname = "scene.solve_physics"
    bl_label = "Solve"
    
    def execute(self, context):
#################################################################################################################################
        class Physics_Object:
            def __init__(self, obj, mesh):
                self.obj = obj
                self.mesh = mesh
                self.point_count = len(mesh.vertices)
                self.edge_count = len(mesh.edges)
                self.tet_count = len(mesh.polygons)
                np.set_printoptions(threshold=np.inf, linewidth=np.inf)
                # Initialize point properties
                self.point_coordinates = np.array([co for vertex in mesh.vertices for co in vertex.co], dtype=np.float32).flatten()
                self.point_initial_coordiantes = self.point_coordinates.copy()
                self.point_prev_coordinates = self.point_coordinates.copy()
                self.point_velocities = np.zeros(3 * self.point_count, dtype=np.float32)
                self.point_inv_masses = np.full(self.point_count, 50000000.0, dtype=np.float32)
                self.constraint_toggle_list = None
                
                self.is_pinned = np.full(self.point_count, 0, dtype=np.int32)
                if obj.vertex_groups:
                    try:
                        pin_group = obj.vertex_groups["Pin_Group"]
                        for i in range(self.point_count):
                            try:
                                if(pin_group.weight(i) > 0.5):
                                    self.is_pinned[i] = 1
                                else:
                                    self.is_pinned[i] = 0
                            except:
                                self.is_pinned[i] = 0
                            finally:
                                pass
                    except:
                        pass
                    finally:
                        pass
                else:
                    print("Object has no vertex groups.")   
                
                self.pin_coordinates = np.copy(self.point_coordinates)
                self.update_pin_coordinates()
                self.mesh_type = 0 
                # None = 0
                # Tet = 1
                # Cloth = 2
                # Non manifold = 3
                # Static = 4
                phys_type = self.obj.phys_type
                if phys_type == "NON":
                    self.mesh_type = 0
                elif phys_type == "TET":
                    self.mesh_type = 1
                elif phys_type == "CLOTH":
                    self.mesh_type = 2
                elif phys_type == "MAN":
                    self.mesh_type = 3
                elif phys_type == "STAT":
                    self.mesh_type = 4
                else:
                    print("Unknown phys_type:", phys_type)
                
                if self.mesh_type == 1:
                    self.constraint_toggle_list = [True, True, False]
                elif self.mesh_type == 2:
                    self.constraint_toggle_list = [True, False, True]
                elif self.mesh_type == 3:
                    self.constraint_toggle_list = [True, False, False]
                else:
                    self.constraint_toggle_list = [False, False, False]
                # Initialize edge properties
                if self.constraint_toggle_list[0]:
                    self.edge_rest_lengths = np.array([self.calculate_edge_length(edge) for edge in mesh.edges], dtype=np.float32)
                    self.edge_tension_compliances = np.full(self.edge_count, self.obj.edge_tens_comp, dtype=np.float32)
                    self.edge_compression_compliances = np.full(self.edge_count, self.obj.edge_comp_comp, dtype=np.float32)
                    self.edge_vertex_IDs = np.array([(edge.vertices[0], edge.vertices[1]) for edge in mesh.edges], dtype=np.int32).flatten()
                    self.edge_dampings = np.full(self.edge_count, self.obj.edge_damp, dtype=np.float32)
                else:
                    self.edge_rest_lengths = np.array([], dtype=np.float32)
                    self.edge_tension_compliances = np.array([], dtype=np.float32)
                    self.edge_compression_compliances = np.array([], dtype=np.float32)
                    self.edge_vertex_IDs = np.array([], dtype=np.int32)
                    self.edge_dampings = np.array([], dtype=np.float32)
                    self.edge_count = 0

                # Initialize tetrahedron properties
                if self.constraint_toggle_list[1]:
                    self.point_inv_masses = np.zeros(self.point_count, dtype=np.float32)
                    self.tet_rest_volumes = np.zeros(self.tet_count, dtype=np.float32)
                    self.vol_compliances = np.full(self.tet_count, self.obj.vol_comp, dtype=np.float32)
                    self.tet_vertex_IDs = np.array([(poly.vertices[0], poly.vertices[1], poly.vertices[2], poly.vertices[3]) for poly in self.mesh.polygons], dtype=np.int32).flatten()
                    self.init_physics()
                    self.trilist = self.get_surface_triangles()
                    self.surface_tris = np.array(self.trilist, dtype=np.int32).flatten()
                    self.surface_tri_count = len(self.surface_tris)/3
                    self.vol_dampings = np.full(self.tet_count, self.obj.vol_damp, dtype=np.float32)
                    self.tri_vertex_IDs = np.array([], dtype=np.int32)
                    self.tri_count = 0
                else:
                    self.tet_rest_volumes = np.array([], dtype=np.float32)
                    self.vol_compliances = np.array([], dtype=np.float32)
                    self.tet_vertex_IDs = np.array([], dtype=np.int32)
                    self.trilist = np.array([], dtype=np.int32)
                    self.surface_tris = np.array([], dtype=np.int32)
                    self.vol_dampings = np.array([], dtype=np.float32)
                    self.surface_tri_count = 0
                    self.tet_count = 0
                    self.tri_vertex_IDs = np.array([(poly.vertices[0], poly.vertices[1], poly.vertices[2]) for poly in self.mesh.polygons], dtype=np.int32).flatten()
                    self.tri_count = len(mesh.polygons)
                
                # Initialize bending properties
                if self.constraint_toggle_list[2]:
                    self.bending_edge_vertex_IDs = self.find_bending_edges()
                    self.bending_edge_count = int(len(self.bending_edge_vertex_IDs) / 2)
                    self.bending_edge_rest_lengths = np.array([self.calculate_edge_length_generic(self.bending_edge_vertex_IDs[i*2], self.bending_edge_vertex_IDs[i*2 + 1]) for i in range(self.bending_edge_count)], dtype=np.float32)
                    self.bending_edge_vertex_IDs = np.array(self.bending_edge_vertex_IDs, dtype=np.int32).flatten()
                    self.bending_compliances = np.full(self.bending_edge_count, self.obj.bend_comp, dtype=np.float32)
                    self.bending_dampings = np.full(self.bending_edge_count, self.obj.bend_damp, dtype=np.float32)
                else:
                    self.bending_edge_vertex_IDs = np.array([], dtype=np.int32)
                    self.bending_edge_count = 0
                    self.bending_edge_rest_lengths = np.array([], dtype=np.float32)
                    self.bending_edge_vertex_IDs = np.array([], dtype=np.int32)
                    self.bending_compliances = np.array([], dtype=np.float32)
                    self.bending_dampings = np.array([], dtype=np.float32)
                    
                self.vertex_collider_radius_multiplier = 1.0
                self.vertex_collider_radius_multiplier = self.obj.vert_rad_mult
                
                #Read Vertex Groups
                if obj.vertex_groups:
                    try:
                        edge_tension_group = obj.vertex_groups["Phys_Edge_Tension"]
                        for i in range(self.edge_count):
                            weight1 = 0.0
                            weight2 = 0.0
                            try:
                                weight1 = edge_tension_group.weight(self.edge_vertex_IDs[i * 2])
                            except:
                                weight1 = 0.0
                            finally:
                                pass
                            try:
                                weight2 = edge_tension_group.weight(self.edge_vertex_IDs[i * 2 + 1])
                            except:
                                weight2 = 0.0
                            finally:
                                pass
                            avg_weight = (weight1+weight2)/2.0
                            self.edge_tension_compliances[i] *= avg_weight
                    except:
                        pass
                    finally:
                        pass
                
                    try:
                        edge_compression_group = obj.vertex_groups["Phys_Edge_Compression"]
                        for i in range(self.edge_count):
                            weight1 = 0.0
                            weight2 = 0.0
                            try:
                                weight1 = edge_compression_group.weight(self.edge_vertex_IDs[i * 2])
                            except:
                                weight1 = 0.0
                            finally:
                                pass
                            try:
                                weight2 = edge_compression_group.weight(self.edge_vertex_IDs[i * 2 + 1])
                            except:
                                weight2 = 0.0
                            finally:
                                pass
                            avg_weight = (weight1+weight2)/2.0
                            self.edge_compression_compliances[i] *= avg_weight
                    except:
                        pass
                    finally:
                        pass
                    
                    try:
                        vol_compliance_group = obj.vertex_groups["Phys_Vol_Compliance"]
                        for i in range(self.tet_count):
                            weight1 = 0.0
                            weight2 = 0.0
                            weight3 = 0.0
                            weight4 = 0.0
                            try:
                                weight1 = vol_compliance_group.weight(self.tet_vertex_IDs[i * 4])
                            except:
                                weight1 = 0.0
                            finally:
                                pass
                            try:
                                weight2 = vol_compliance_group.weight(self.tet_vertex_IDs[i * 4 + 1])
                            except:
                                weight2 = 0.0
                            finally:
                                pass
                            try:
                                weight3 = vol_compliance_group.weight(self.tet_vertex_IDs[i * 4 + 2])
                            except:
                                weight3 = 0.0
                            finally:
                                pass
                            try:
                                weight4 = vol_compliance_group.weight(self.tet_vertex_IDs[i * 4 + 3])
                            except:
                                weight4 = 0.0
                            finally:
                                pass
                            avg_weight = (weight1+weight2+weight3+weight4)/4.0
                            self.vol_compliances[i] *= avg_weight
                    except:
                        pass
                    finally:
                        pass
                
                    try:
                        edge_damping_group = obj.vertex_groups["Phys_Edge_Damping"]
                        for i in range(self.edge_count):
                            weight1 = 0.0
                            weight2 = 0.0
                            try:
                                weight1 = edge_damping_group.weight(self.edge_vertex_IDs[i * 2])
                            except:
                                weight1 = 0.0
                            finally:
                                pass
                            try:
                                weight2 = edge_damping_group.weight(self.edge_vertex_IDs[i * 2 + 1])
                            except:
                                weight2 = 0.0
                            finally:
                                pass
                            avg_weight = (weight1+weight2)/2.0
                            self.edge_dampings[i] *= avg_weight
                    except:
                        pass
                    finally:
                        pass
                    
                    try:
                        vol_damping_group = obj.vertex_groups["Phys_Vol_Damping"]
                        for i in range(self.tet_count):
                            weight1 = 0.0
                            weight2 = 0.0
                            weight3 = 0.0
                            weight4 = 0.0
                            try:
                                weight1 = vol_damping_group.weight(self.tet_vertex_IDs[i * 4])
                            except:
                                weight1 = 0.0
                            finally:
                                pass
                            try:
                                weight2 = vol_damping_group.weight(self.tet_vertex_IDs[i * 4 + 1])
                            except:
                                weight2 = 0.0
                            finally:
                                pass
                            try:
                                weight3 = vol_damping_group.weight(self.tet_vertex_IDs[i * 4 + 2])
                            except:
                                weight3 = 0.0
                            finally:
                                pass
                            try:
                                weight4 = vol_damping_group.weight(self.tet_vertex_IDs[i * 4 + 3])
                            except:
                                weight4 = 0.0
                            finally:
                                pass
                            avg_weight = (weight1+weight2+weight3+weight4)/4.0
                            self.vol_dampings[i] *= avg_weight
                    except:
                        pass
                    finally:
                        pass
            
            def cache_mesh(self, frame):
                bpy.ops.object.select_all(action='DESELECT')
                self.obj.select_set(True)
                directory = bpy.path.abspath(f"//alembic_sequence/{self.obj.name}/")
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filepath = os.path.join(directory, f"{self.obj.name}_{frame:05}.abc")
                bpy.ops.wm.alembic_export(filepath=filepath, selected=True)
                
                    
            def update_pin_coordinates(self):
                obj_ref_name = self.obj.name
                if obj_ref_name not in bpy.data.objects:
                    #print(f"Object '{obj_ref_name}' not found.")
                    return
                
                obj_ref = bpy.data.objects[obj_ref_name].reference_object
                if obj_ref is None:
                    #print(f"Reference object for '{obj_ref_name}' is not defined.")
                    return
                   
                depsgraph = bpy.context.evaluated_depsgraph_get()
                bm = bmesh.new()
                bm.from_object(obj_ref, depsgraph)
                self.pin_coordinates = np.array([co for vertex in bm.verts for co in vertex.co], dtype=np.float32).flatten()
                
            def reset_mesh(self):
                self.point_coordinates = self.point_initial_coordiantes.copy()
                self.update_mesh()
                
            def calculate_edge_length(self, edge):
                index_a, index_b = edge.vertices[0], edge.vertices[1]
                co_a = np.array(self.mesh.vertices[index_a].co)
                co_b = np.array(self.mesh.vertices[index_b].co)
                return np.linalg.norm(co_b - co_a)
            
            def calculate_edge_length_generic(self, p1, p2):
                co_a = np.array(self.mesh.vertices[p1].co)
                co_b = np.array(self.mesh.vertices[p2].co)
                return np.linalg.norm(co_b - co_a)

            def update_mesh(self):
                for i, co in enumerate(self.point_coordinates.reshape((-1, 3))):
                    self.mesh.vertices[i].co = co
                    
            def get_tet_volume(self, nr):
                id0 = self.tet_vertex_IDs[4 * nr]
                id1 = self.tet_vertex_IDs[4 * nr + 1]
                id2 = self.tet_vertex_IDs[4 * nr + 2]
                id3 = self.tet_vertex_IDs[4 * nr + 3]

                temp = np.zeros((4, 3), dtype=np.float32)

                self.vec_set_diff(temp, 0, self.point_coordinates, id1, self.point_coordinates, id0)
                self.vec_set_diff(temp, 1, self.point_coordinates, id2, self.point_coordinates, id0)
                self.vec_set_diff(temp, 2, self.point_coordinates, id3, self.point_coordinates, id0)
                self.vec_set_cross(temp, 3, temp, 0, temp, 1)

                return np.dot(temp[3], temp[2]) / 6.0

            def init_physics(self):
                self.point_inv_masses.fill(0.0)
                self.tet_rest_volumes.fill(0.0)

                for i in range(self.tet_count):
                    vol = self.get_tet_volume(i)
                    self.tet_rest_volumes[i] = vol
                    p_inv_mass = 1.0 / (vol / 4.0) if vol > 0.0 else 0.0
                    self.point_inv_masses[self.tet_vertex_IDs[4 * i]] += p_inv_mass
                    self.point_inv_masses[self.tet_vertex_IDs[4 * i + 1]] += p_inv_mass
                    self.point_inv_masses[self.tet_vertex_IDs[4 * i + 2]] += p_inv_mass
                    self.point_inv_masses[self.tet_vertex_IDs[4 * i + 3]] += p_inv_mass

                for i in range(self.edge_count):
                    id0 = self.edge_vertex_IDs[2 * i]
                    id1 = self.edge_vertex_IDs[2 * i + 1]
                    self.edge_rest_lengths[i] = np.sqrt(np.sum((self.point_coordinates[id1 * 3: id1 * 3 + 3] -
                                                                 self.point_coordinates[id0 * 3: id0 * 3 + 3]) ** 2))

            def vec_set_diff(self, result, result_idx, vec_a, idx_a, vec_b, idx_b):
                result[result_idx] = vec_a[idx_a * 3: idx_a * 3 + 3] - vec_b[idx_b * 3: idx_b * 3 + 3]

            def vec_set_cross(self, result, result_idx, vec_a, idx_a, vec_b, idx_b):
                result[result_idx] = np.cross(vec_a[idx_a], vec_b[idx_b])
                
            def get_tetrahedron_triangles(self):
                triangles = []
                start_idx = 0
                
                end_idx = start_idx + self.tet_count * 4
                tet_vertex_indices = self.tet_vertex_IDs[start_idx:end_idx].reshape(-1, 4)
                for tet_indices in tet_vertex_indices:
                    triangles.extend([[tet_indices[0], tet_indices[1], tet_indices[2]],
                                      [tet_indices[0], tet_indices[1], tet_indices[3]],
                                      [tet_indices[1], tet_indices[2], tet_indices[3]],
                                      [tet_indices[0], tet_indices[2], tet_indices[3]]])
                start_idx = end_idx
                return triangles
            
            def get_surface_triangles(self):
                
                triangles = self.get_tetrahedron_triangles()
                
                triangle_sets = []
                
                for t in triangles:
                    triangle_sets.append(frozenset(t))
                    
                    
                grouped_sets = []
                set_map = {}

                for fs in triangle_sets:
                    # Convert the set to a frozenset for immutability
                    if fs in set_map:
                        grouped_sets[set_map[fs]].append(fs)
                    else:
                        set_map[fs] = len(grouped_sets)
                        grouped_sets.append([fs])
                
                filtered = [group for group in grouped_sets if len(group) == 1]
                
                filt_list = [list(s[0]) for s in filtered]
                return filt_list
            
            def find_bending_edges(self):
                def find_neighbors(triangle_index):
                    neighbors = []
                    for poly in self.mesh.polygons:
                        if len(poly.vertices) == 3 and poly.index != triangle_index:
                            shared_verts = set(self.mesh.polygons[triangle_index].vertices) & set(poly.vertices)
                            if len(shared_verts) == 2:
                                neighbors.append(poly.index)
                    return neighbors
                
                triangle_neighbors = []
                
                for i, poly in enumerate(self.mesh.polygons):
                    if len(poly.vertices) == 3:  # Check if polygon is a triangle
                        triangle_neighbors.append([])  # Append an empty list for this triangle's neighbors
                        neighbors = find_neighbors(i)
                        for neighbor_index in neighbors:
                            triangle_verts = set(self.mesh.polygons[i].vertices)
                            neighbor_verts = set(self.mesh.polygons[neighbor_index].vertices)
                            # Subtract the union of sets with intersection of sets
                            result = (triangle_verts | neighbor_verts) - (triangle_verts & neighbor_verts)
                            triangle_neighbors[-1].append(result)  # Add the result to the list
                result=[]
                for e in triangle_neighbors:
                    for e1 in e:
                        result.append(e1)
                
                unique_sets = set(frozenset(s) for s in result)
                unique_list_of_sets = [set(fs) for fs in unique_sets]
                unique_list_of_lists = [list(s) for s in unique_list_of_sets]
                
                output=[]
                
                for e in unique_list_of_sets:
                    for e1 in e:
                        output.append(e1)
                        
                return output
                
#################################################################################################################################                
             

        class Physics_Environment:
            def __init__(self, phys_objs):
                self.phys_objs = phys_objs
                self.mesh_count = len(self.phys_objs)
                # Point counts
                self.point_counts = np.array([obj.point_count for obj in phys_objs], dtype=np.int32)
                self.point_count = np.sum(self.point_counts)
                # Edge counts
                self.edge_counts = np.array([obj.edge_count for obj in phys_objs], dtype=np.int32)
                self.edge_count = np.sum(self.edge_counts)
                # Tet counts
                self.tet_counts = np.array([obj.tet_count for obj in phys_objs], dtype=np.int32)
                self.tet_count = np.sum(self.tet_counts)
                # Tris
                self.surface_tri_counts = np.array([obj.surface_tri_count for obj in phys_objs], dtype=np.int32).flatten()
                self.surface_tris = np.concatenate([obj.surface_tris + np.sum(self.point_counts[:i]) for i, obj in enumerate(phys_objs)]).flatten()
                # 
                self.point_coordinates = np.concatenate([obj.point_coordinates for obj in phys_objs]).flatten()
                self.point_prev_coordinates = np.concatenate([obj.point_prev_coordinates for obj in phys_objs]).flatten()
                self.point_velocities = np.concatenate([obj.point_velocities for obj in phys_objs]).flatten()
                self.point_inv_masses = np.concatenate([obj.point_inv_masses for obj in phys_objs]).flatten()
                # 
                self.edge_rest_lengths = np.concatenate([obj.edge_rest_lengths for obj in phys_objs]).flatten()
                self.edge_tension_compliances = np.concatenate([obj.edge_tension_compliances for obj in phys_objs]).flatten()
                self.edge_compression_compliances = np.concatenate([obj.edge_compression_compliances for obj in phys_objs]).flatten()
                self.edge_vertex_IDs = np.concatenate([obj.edge_vertex_IDs + np.sum(self.point_counts[:i]) for i, obj in enumerate(phys_objs)]).flatten()
                # 
                self.tet_rest_volumes = np.concatenate([obj.tet_rest_volumes for obj in phys_objs]).flatten()
                self.vol_compliances = np.concatenate([obj.vol_compliances for obj in phys_objs]).flatten()
                self.tet_vertex_IDs = np.concatenate([obj.tet_vertex_IDs + np.sum(self.point_counts[:i]) for i, obj in enumerate(phys_objs)]).flatten()
                
                self.gravity = np.array([0.0, 0.0, -10.0], dtype=np.float32)
                self.gravity[0] = bpy.data.scenes["Scene"].physics_settings.gravity[0]
                self.gravity[1] = bpy.data.scenes["Scene"].physics_settings.gravity[1]
                self.gravity[2] = bpy.data.scenes["Scene"].physics_settings.gravity[2]
                
                self.surface_tris = np.concatenate([obj.surface_tris + np.sum(self.point_counts[:i]) for i, obj in enumerate(phys_objs)]).flatten()
                self.surface_tri_counts = np.array([obj.surface_tri_count for obj in phys_objs], dtype=np.int32).flatten()
                self.surface_tri_count = np.sum(self.surface_tri_counts)
                
                self.bending_edge_vertex_IDs = np.concatenate([obj.bending_edge_vertex_IDs + np.sum(self.point_counts[:i]) for i, obj in enumerate(phys_objs)]).flatten()
                self.bending_edge_counts = np.array([obj.bending_edge_count for obj in phys_objs], dtype=np.int32)
                self.bending_edge_count = np.sum(self.bending_edge_counts)
                self.bending_edge_rest_lengths = np.concatenate([obj.bending_edge_rest_lengths for obj in phys_objs]).flatten()
                self.bending_compliances = np.concatenate([obj.bending_compliances for obj in phys_objs]).flatten()
                
                self.pin_coordinates = np.concatenate([obj.pin_coordinates for obj in phys_objs]).flatten()
                self.is_pinned = np.concatenate([obj.is_pinned for obj in phys_objs]).flatten()
                
                self.vol_dampings = np.concatenate([obj.vol_dampings for obj in phys_objs]).flatten()
                self.edge_dampings = np.concatenate([obj.edge_dampings for obj in phys_objs]).flatten()
                self.bending_dampings = np.concatenate([obj.bending_dampings for obj in phys_objs]).flatten()
                
                self.tri_vertex_IDs = np.concatenate([obj.tri_vertex_IDs + np.sum(self.point_counts[:i]) for i, obj in enumerate(phys_objs)]).flatten()
                self.tri_counts = np.array([obj.tri_count for obj in phys_objs], dtype=np.int32)
                self.tri_count = np.sum(self.tri_counts)
                
                self.mesh_types = np.array([obj.mesh_type for obj in phys_objs], dtype=np.int32)
                
                self.vertex_collider_radius_multiplier = np.array([obj.vertex_collider_radius_multiplier for obj in phys_objs], dtype=np.float32).flatten()
                
            def update_point_coordinates(self):
                start_idx = 0
                for obj in self.phys_objs:
                    end_idx = start_idx + obj.point_count * 3
                    obj.point_coordinates[:] = self.point_coordinates[start_idx:end_idx]
                    obj.point_velocities[:] = self.point_velocities[start_idx:end_idx]
                    start_idx = end_idx
                    
            def update_pin_coordinates(self):
                for obj in self.phys_objs:
                    obj.update_pin_coordinates()
                self.pin_coordinates = np.concatenate([obj.pin_coordinates for obj in self.phys_objs]).flatten()
                
            def reset_meshes(self):
                for obj in self.phys_objs:
                    obj.reset_mesh()
                
            def cache_meshes(self, frame):
                for obj in self.phys_objs:
                    obj.cache_mesh(frame)
                
            def solve_frame_external(self):
                # Load the external C library
                #"D:/Files/Projects/Programming/Github/Blender-GPU-Physics/build/libCUDAXPBD.so"
                lib_path = bpy.data.scenes["Scene"].physics_settings.solver_path
                lib = ctypes.CDLL(lib_path)

                # Extract numpy arrays from Physics_Environment attributes
                point_coordinates_ptr = self.point_coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                point_prev_coordinates_ptr = self.point_prev_coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                point_velocities_ptr = self.point_velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                point_inv_masses_ptr = self.point_inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                edge_rest_lengths_ptr = self.edge_rest_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                edge_tension_compliances_ptr = self.edge_tension_compliances.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                edge_compression_compliances_ptr = self.edge_compression_compliances.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                tet_rest_volumes_ptr = self.tet_rest_volumes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                vol_compliances_ptr = self.vol_compliances.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

                gravity_ptr = self.gravity.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                point_counts_ptr = self.point_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                edge_counts_ptr = self.edge_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                tet_counts_ptr = self.tet_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                edge_vertex_IDs_ptr = self.edge_vertex_IDs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                tet_vertex_IDs_ptr = self.tet_vertex_IDs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                tri_vertex_IDs_ptr = self.tri_vertex_IDs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                tri_counts_ptr = self.tri_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                
                surface_tris_ptr = self.surface_tris.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                surface_tri_counts_ptr = self.surface_tri_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                
                bending_edge_vertex_IDs_ptr = self.bending_edge_vertex_IDs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                bending_edge_counts_ptr = self.bending_edge_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                bending_edge_rest_lengths_ptr = self.bending_edge_rest_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                bending_compliances_ptr = self.bending_compliances.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                pin_coordinates_ptr = self.pin_coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                is_pinned_ptr = self.is_pinned.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                
                vol_dampings_ptr = self.vol_dampings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                edge_dampings_ptr = self.edge_dampings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                bending_dampings_ptr = self.bending_dampings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                mesh_types_ptr = self.mesh_types.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                
                vertex_collider_radius_multiplier_ptr = self.vertex_collider_radius_multiplier.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                # Specify the parameter and return types of the C function
                lib.solve_frame.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # point_coordinates
                    ctypes.POINTER(ctypes.c_float),  # point_prev_coordinates
                    ctypes.POINTER(ctypes.c_float),  # point_velocities
                    ctypes.POINTER(ctypes.c_float),  # point_inv_masses
                    ctypes.POINTER(ctypes.c_float),  # edge_rest_lengths
                    ctypes.POINTER(ctypes.c_float),  # edge_tension_compliances
                    ctypes.POINTER(ctypes.c_float),  # edge_compression_compliances
                    ctypes.POINTER(ctypes.c_float),  # tet_rest_volumes
                    ctypes.POINTER(ctypes.c_float),  # vol_compliances
                    ctypes.POINTER(ctypes.c_float),  # bending_edge_rest_lengths
                    ctypes.POINTER(ctypes.c_float),  # bending_compliances
                    ctypes.POINTER(ctypes.c_float),  # edge_dampings
                    ctypes.POINTER(ctypes.c_float),  # vol_dampings
                    ctypes.POINTER(ctypes.c_float),  # bending_dampings
                    ctypes.POINTER(ctypes.c_float),  # gravity
                    ctypes.POINTER(ctypes.c_int),    # point_counts
                    ctypes.POINTER(ctypes.c_int),    # edge_counts
                    ctypes.POINTER(ctypes.c_int),    # tet_counts
                    ctypes.POINTER(ctypes.c_int),    # tri_counts
                    ctypes.POINTER(ctypes.c_int),    # bending_edge_counts
                    ctypes.POINTER(ctypes.c_int),    # edge_vertex_IDs
                    ctypes.POINTER(ctypes.c_int),    # tet_vertex_IDs
                    ctypes.POINTER(ctypes.c_int),    # tri_vertex_IDs
                    ctypes.POINTER(ctypes.c_int),    # surface_tris
                    ctypes.POINTER(ctypes.c_int),    # surface_tri_counts
                    ctypes.POINTER(ctypes.c_int),    # bending_edge_vertex_IDs
                    ctypes.POINTER(ctypes.c_int),    # mesh_types
                    ctypes.POINTER(ctypes.c_float),    # pin_coords
                    ctypes.POINTER(ctypes.c_int),    # is_pinned
                    ctypes.POINTER(ctypes.c_float),    # vertex collider radius multiplier
                    ctypes.c_int,  # surface_tri_count
                    ctypes.c_int,  # point_count
                    ctypes.c_int,  # edge_count
                    ctypes.c_int,  # tet_counta
                    ctypes.c_int,  # tri_count
                    ctypes.c_int,  # mesh_count
                    ctypes.c_int,  # bending_edge_count
                    ctypes.c_int,  # frequency
                    ctypes.c_int,  # sim_res
                    ctypes.c_float, # floor_friction
                    ctypes.c_float # macro_cell_size
                ]
                lib.solve_frame.restype = None  # Assuming the function has no return value

                # Extract numpy arrays from Physics_Environment attributes
                # ... (previous code)
                total_iter = bpy.data.scenes["Scene"].physics_settings.phys_iter
                fps = bpy.data.scenes["Scene"].physics_settings.phys_freq
                substeps = int(total_iter / fps)
                # Call the external C function
                lib.solve_frame(
                    point_coordinates_ptr,
                    point_prev_coordinates_ptr,
                    point_velocities_ptr,
                    point_inv_masses_ptr,
                    edge_rest_lengths_ptr,
                    edge_tension_compliances_ptr,
                    edge_compression_compliances_ptr,
                    tet_rest_volumes_ptr,
                    vol_compliances_ptr,
                    bending_edge_rest_lengths_ptr,
                    bending_compliances_ptr,
                    edge_dampings_ptr,
                    vol_dampings_ptr,
                    bending_dampings_ptr,
                    gravity_ptr,
                    point_counts_ptr,
                    edge_counts_ptr,
                    tet_counts_ptr,
                    tri_counts_ptr,
                    bending_edge_counts_ptr,
                    edge_vertex_IDs_ptr,
                    tet_vertex_IDs_ptr,
                    tri_vertex_IDs_ptr,
                    surface_tris_ptr,
                    surface_tri_counts_ptr,
                    bending_edge_vertex_IDs_ptr,
                    mesh_types_ptr,
                    pin_coordinates_ptr,
                    is_pinned_ptr,
                    vertex_collider_radius_multiplier_ptr,
                    self.surface_tri_count,
                    self.point_count,
                    self.edge_count,
                    self.tet_count,
                    self.tri_count,
                    self.mesh_count,
                    self.bending_edge_count,
                    fps,  
                    substeps,
                    bpy.data.scenes["Scene"].physics_settings.floor_friction,
                    0.3
                )
#################################################################################################################################
        collection_name = bpy.data.scenes["Scene"].physics_settings.physics_collection.name
        physics_objects = []
        
        for i, obj in enumerate(bpy.data.collections[collection_name].all_objects):
            if obj.type == 'MESH':
                physics_obj = Physics_Object(obj, obj.data)
                physics_objects.append(physics_obj)

        physics_environment = Physics_Environment(physics_objects)

        counter = 0
        sim_length = (bpy.data.scenes["Scene"].physics_settings.frame_end - bpy.data.scenes["Scene"].physics_settings.frame_start)
        def run_sim():
            nonlocal counter
            physics_environment.update_pin_coordinates()
            physics_environment.solve_frame_external()
            physics_environment.update_point_coordinates()
            
            for physics_obj in physics_environment.phys_objs:
                physics_obj.update_mesh()
            physics_environment.cache_meshes(bpy.context.scene.frame_current)
            counter += 1
            bpy.context.scene.frame_current += 1
            
            progress = (counter) / sim_length
            update_progress_bar(progress)
                
            if counter >= sim_length:
                physics_environment.reset_meshes()
                bpy.context.scene.frame_current = bpy.data.scenes["Scene"].physics_settings.frame_start
                return None
            
            return 0.001

        def launch_sim():
            bpy.context.scene.frame_current = bpy.data.scenes["Scene"].physics_settings.frame_start
            bpy.app.timers.register(run_sim)
            
        launch_sim()


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
