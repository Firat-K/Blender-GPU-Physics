import numpy as np
import bpy
import time
exec(bpy.data.texts['Physics_Object'].as_string())

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
        self.edge_constraint_colors = np.concatenate([obj.edge_constraint_colors for obj in phys_objs]).flatten()
        self.edge_constraint_max_color = np.amax(self.edge_constraint_colors)
        # 
        self.tet_rest_volumes = np.concatenate([obj.tet_rest_volumes for obj in phys_objs]).flatten()
        self.vol_compliances = np.concatenate([obj.vol_compliances for obj in phys_objs]).flatten()
        self.vol_constraint_colors = np.concatenate([obj.vol_constraint_colors for obj in phys_objs]).flatten()
        self.tet_vertex_IDs = np.concatenate([obj.tet_vertex_IDs + np.sum(self.point_counts[:i]) for i, obj in enumerate(phys_objs)]).flatten()
        self.vol_constraint_max_color = np.amax(self.vol_constraint_colors)
        
        self.gravity = np.array([0.0, 0.0, -10.0], dtype=np.float32)
        
        self.surface_tris = np.array([obj.surface_tris for obj in phys_objs], dtype=np.int32).flatten()
        self.surface_tri_counts = np.array([obj.surface_tri_count for obj in phys_objs], dtype=np.int32).flatten()
        self.surface_tri_count = np.sum(self.surface_tri_counts)
        
    def update_point_coordinates(self):
        start_idx = 0
        for obj in self.phys_objs:
            end_idx = start_idx + obj.point_count * 3
            obj.point_coordinates[:] = self.point_coordinates[start_idx:end_idx]
            obj.point_velocities[:] = self.point_velocities[start_idx:end_idx]
            start_idx = end_idx
        
    def solve_frame_external(self):
        # Load the external C library
        lib_path = "D:/Files/Projects/Programming/TiddyEngine/build/libCUDAXPBD.so"
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
        edge_constraint_colors_ptr = self.edge_constraint_colors.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        edge_vertex_IDs_ptr = self.edge_vertex_IDs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        vol_constraint_colors_ptr = self.vol_constraint_colors.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        tet_vertex_IDs_ptr = self.tet_vertex_IDs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        surface_tris_ptr = self.surface_tris.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        surface_tri_counts_ptr = self.surface_tri_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
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
            ctypes.POINTER(ctypes.c_float),  # gravity
            ctypes.POINTER(ctypes.c_int),    # point_counts
            ctypes.POINTER(ctypes.c_int),    # edge_counts
            ctypes.POINTER(ctypes.c_int),    # tet_counts
            ctypes.POINTER(ctypes.c_int),    # edge_constraint_colors
            ctypes.POINTER(ctypes.c_int),    # edge_vertex_IDs
            ctypes.POINTER(ctypes.c_int),    # vol_constraint_colors
            ctypes.POINTER(ctypes.c_int),    # tet_vertex_IDs
            ctypes.POINTER(ctypes.c_int),    # surface_tris
            ctypes.POINTER(ctypes.c_int),    # surface_tri_counts
            ctypes.c_int,  # surface_tri_count
            ctypes.c_int,  # point_count
            ctypes.c_int,  # edge_count
            ctypes.c_int,  # tet_count
            ctypes.c_int,  # mesh_count
            ctypes.c_int,  # edge_constraint_color_count
            ctypes.c_int,  # tet_constraint_color_count
            ctypes.c_int,  # frequency
            ctypes.c_int   # sim_res
        ]
        lib.solve_frame.restype = None  # Assuming the function has no return value

        # Extract numpy arrays from Physics_Environment attributes
        # ... (previous code)

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
            gravity_ptr,
            point_counts_ptr,
            edge_counts_ptr,
            tet_counts_ptr,
            edge_constraint_colors_ptr,
            edge_vertex_IDs_ptr,
            vol_constraint_colors_ptr,
            tet_vertex_IDs_ptr,
            surface_tris_ptr,
            surface_tri_counts_ptr,
            self.surface_tri_count,
            self.point_count,
            self.edge_count,
            self.tet_count,
            self.mesh_count,
            self.edge_constraint_max_color,
            self.vol_constraint_max_color,
            60,  # Example value for frequency
            15  # Example value for sim_res
        )

# Assuming you have a collection named "Physics_Test" with objects inside it
collection_name = "Physics_Test"
physics_objects = []

# Iterate through objects in the collection and create Physics_Object instances
for obj in bpy.data.collections[collection_name].all_objects:
    if obj.type == 'MESH':
        physics_obj = Physics_Object(obj, obj.data, 100.0, 10.0, 0.0)  # You may adjust the parameters accordingly
        physics_objects.append(physics_obj)

# Create a Physics_Environment instance with the list of Physics_Objects
physics_environment = Physics_Environment(physics_objects)

counter = 0

def run_sim():
    global physics_environment
    global counter
    physics_environment.solve_frame_external()
    physics_environment.update_point_coordinates()
    for physics_obj in physics_environment.phys_objs:
        physics_obj.update_mesh()
    counter += 1
    if counter == 60:
        return None
    return 0.01

bpy.app.timers.register(run_sim)


