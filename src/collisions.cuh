#ifndef COLLISIONS_CUH_
#define COLLISIONS_CUH_

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "vecmath.cuh"
#define DLLEXPORT extern "C" __declspec(dllexport)

__global__ void tet_tet_mark_inside(
    float* point_coordinates,
    int* tet_vertex_IDs,
    int* point_counts,
    int* tet_counts,
    int* point_in_out,
    int mesh_ID_0, //collider mesh
    int mesh_ID_1 //target mesh
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int mesh_1_point_offset = get_mesh_offset(point_counts, mesh_ID_1);
    int mesh_0_point_offset = get_mesh_offset(point_counts, mesh_ID_0);
    int mesh_1_tet_offset = get_mesh_offset(tet_counts, mesh_ID_1);
    int mesh_0_tet_offset = get_mesh_offset(tet_counts, mesh_ID_0);
    if(idx < point_counts[mesh_ID_1])
    {
        float* target_point = &point_coordinates[(mesh_1_point_offset + idx) * 3];
        for(int i = 0; i < tet_counts[mesh_ID_0]; i++)
        {
            float* p0 = &point_coordinates[(tet_vertex_IDs[((mesh_0_tet_offset + i) * 4) + 0]) * 3];
            float* p1 = &point_coordinates[(tet_vertex_IDs[((mesh_0_tet_offset + i) * 4) + 1]) * 3];
            float* p2 = &point_coordinates[(tet_vertex_IDs[((mesh_0_tet_offset + i) * 4) + 2]) * 3];
            float* p3 = &point_coordinates[(tet_vertex_IDs[((mesh_0_tet_offset + i) * 4) + 3]) * 3];
            if(is_point_in_tet(p0, p1, p2, p3, target_point))
            {
                point_in_out[mesh_1_point_offset + idx] = 1;
                break;
            }
            else
            {
                point_in_out[mesh_1_point_offset + idx] = 0;
            }
        }
    }
}

__global__ void set_tet_collision_correction_vectors(
    float* point_coordinates,
    int* point_counts,
    int* point_in_out,
    int* edge_vertex_IDs,
    int* edge_counts,
    float* correction_vectors,
    int mesh_ID //target mesh
)
{   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int mesh_point_offset = get_mesh_offset(point_counts, mesh_ID);
    int mesh_edge_offset = get_mesh_offset(edge_counts, mesh_ID);
    if(idx < edge_counts[mesh_ID])
    {
        int id0 = edge_vertex_IDs[(mesh_edge_offset + idx) * 2];
        int id1 = edge_vertex_IDs[(mesh_edge_offset + idx) * 2 + 1];
        if(point_in_out[id0] == 1 && point_in_out[id1] == 0)
        { 
            float correction_vector[3] = {0.0, 0.0, 0.0};
            sub_vectors(&point_coordinates[id1 * 3], &point_coordinates[id0 * 3], correction_vector);
            normalize_vector_to(correction_vector, correction_vector);
            add_vectors(&correction_vectors[id0 * 3], correction_vector, &correction_vectors[id0 * 3]);
            normalize_vector_to(&correction_vectors[id0 * 3], &correction_vectors[id0 * 3]);
        }
        else if(point_in_out[id0] == 0 && point_in_out[id1] == 1)
        {
            float correction_vector[3] = {0.0, 0.0, 0.0};
            sub_vectors(&point_coordinates[id0 * 3], &point_coordinates[id1 * 3], correction_vector);
            normalize_vector_to(correction_vector, correction_vector);
            add_vectors(&correction_vectors[id1 * 3], correction_vector, &correction_vectors[id1 * 3]);
            normalize_vector_to(&correction_vectors[id1 * 3], &correction_vectors[id1 * 3]);
        }
    }
}

__global__ void project_point_tet_mesh(
    float* corrections,
    float* point_coordinates,
    int* point_counts,
    int* surface_triangles,
    int* surface_tri_counts,
    int* point_in_out,
    int mesh_ID_target, //target mesh
    int mesh_ID_collider //collider mesh
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int mesh_target_point_offset = get_mesh_offset(point_counts, mesh_ID_target);
    int mesh_collider_point_offset = get_mesh_offset(point_counts, mesh_ID_collider);
    int mesh_collider_triangle_offset = get_mesh_offset(surface_tri_counts, mesh_ID_collider);
    if(idx < point_counts[mesh_ID_target])
    {
        for(int i = 0; i < surface_tri_counts[mesh_ID_collider]; i++)
        {
            int id0 = mesh_collider_point_offset + surface_triangles[i * 3];
            int id1 = mesh_collider_point_offset + surface_triangles[i * 3 + 1];
            int id2 = mesh_collider_point_offset + surface_triangles[i * 3 + 2];
            int pid = mesh_target_point_offset + idx;

            float intersection[3] = {0.0, 0.0, 0.0};

            if(triangle_ray_tracer(
                &point_coordinates[id0 * 3],
                &point_coordinates[id1 * 3],
                &point_coordinates[id2 * 3],
                &point_coordinates[pid * 3],
                &corrections[pid * 3],
                intersection
            ))
            {
                copy_vector_to(intersection, &point_coordinates[pid * 3]);
                point_in_out[pid] = 0;
                printf("Correction made \n");
                break;
            }
            
        }
    }
}

__global__ void check_if_all_out(
    int* point_in_out,
    int* point_counts,
    int mesh_ID,
    int* out
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int mesh_point_offset = get_mesh_offset(point_counts, mesh_ID);
    if(idx < point_counts[mesh_ID])
    {
        if(point_in_out[mesh_point_offset + idx] == 1)
        {
            *out = 1;
        }
    }
}

#endif /* COLLISIONS_CUH_ */