import bpy
import numpy as np
import ctypes

class Physics_Object:
    def __init__(self, obj, mesh, edge_tension_compliances, edge_compression_compliances, vol_compliances):
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
        self.point_inv_masses = np.zeros(self.point_count, dtype=np.float32)

        # Initialize edge properties
        self.edge_rest_lengths = np.array([self.calculate_edge_length(edge) for edge in mesh.edges], dtype=np.float32)
        self.edge_tension_compliances = np.full(self.edge_count, edge_tension_compliances, dtype=np.float32)
        self.edge_compression_compliances = np.full(self.edge_count, edge_compression_compliances, dtype=np.float32)
        self.edge_constraint_colors = self.assign_edge_colors()
        self.edge_vertex_IDs = np.array([(edge.vertices[0], edge.vertices[1]) for edge in mesh.edges], dtype=np.int32).flatten()
        self.edge_constraint_max_color = np.amax(self.edge_constraint_colors)

        # Initialize tetrahedron properties
        
        self.tet_rest_volumes = np.zeros(self.tet_count, dtype=np.float32)
        self.vol_compliances = np.full(self.tet_count, vol_compliances, dtype=np.float32)
        self.vol_constraint_colors = self.assign_vol_colors()
        self.tet_vertex_IDs = np.array([(poly.vertices[0], poly.vertices[1], poly.vertices[2], poly.vertices[3]) for poly in mesh.polygons], dtype=np.int32).flatten()
        self.init_physics()
        self.tet_constraint_max_color = np.amax(self.vol_constraint_colors)
        
        self.trilist = self.get_surface_triangles()
        self.surface_tris = np.array(self.trilist, dtype=np.int32).flatten()
        self.surface_tri_count = len(self.surface_tris)/3

    def calculate_edge_length(self, edge):
        index_a, index_b = edge.vertices[0], edge.vertices[1]
        co_a = np.array(self.mesh.vertices[index_a].co)
        co_b = np.array(self.mesh.vertices[index_b].co)
        return np.linalg.norm(co_b - co_a)

    def assign_edge_colors(self):
        edge_constraint_colors = np.zeros(self.edge_count, dtype=np.int32)

        # Create a dictionary to store used colors for each vertex index
        vertex_colors = {vertex_index: set() for vertex_index in range(len(self.mesh.vertices))}

        for i, edge in enumerate(self.mesh.edges):
            # Get the colors of neighboring edges for both vertices
            colors_a = vertex_colors[edge.vertices[0]]
            colors_b = vertex_colors[edge.vertices[1]]
            
            available_colors = set(range(1, len(self.mesh.edges) + 1)) - (colors_a | colors_b)
            if available_colors:
                color = min(available_colors)
            else:
                color = 1  # Fallback to color 1 if no available colors (should not happen with connected meshes)

            # Assign the color to the edge
            edge_constraint_colors[i] = color

            # Update the used colors for the vertices
            colors_a.add(color)
            colors_b.add(color)
            
        return edge_constraint_colors

    def assign_vol_colors(self):
        vol_constraint_colors = np.zeros(self.tet_count, dtype=np.int32)

        # Create a dictionary to store used colors for each vertex index
        vertex_colors = {vertex_index: set() for vertex_index in range(len(self.mesh.vertices))}

        for i, poly in enumerate(self.mesh.polygons):
            # Get the colors of neighboring tetrahedra for each vertex
            colors = [vertex_colors[vertex_index] for vertex_index in poly.vertices]

            # Find the first available color for the tetrahedron
            available_colors = set(range(1, len(self.mesh.polygons) + 1)) - set.union(*colors)
            if available_colors:
                color = min(available_colors)
            else:
                color = 1  # Fallback to color 1 if no available colors (should not happen with connected meshes)

            # Assign the color to the tetrahedron
            vol_constraint_colors[i] = color

            # Update the used colors for the vertices
            for vertex_index in poly.vertices:
                vertex_colors[vertex_index].add(color)

        return vol_constraint_colors

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
            
        
        
            