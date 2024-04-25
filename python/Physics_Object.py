import bpy
import numpy as np
import ctypes
import bmesh
import os
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
        obj_ref_name = self.obj.name + "_ref"
        obj_ref = bpy.data.objects.get(obj_ref_name)

        if obj_ref is None:
           print(f"Object '{obj_ref_name}' not found.")
           return
       
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm = bmesh.new()
        bm.from_object(obj_ref, depsgraph)
        self.pin_coordinates = np.array([co for vertex in bm.verts for co in vertex.co], dtype=np.float32).flatten()

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
        
        
            