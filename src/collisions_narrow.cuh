#ifndef COLLISIONS_CUH_
#define COLLISIONS_CUH_

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "vecmath.cuh"
#define DLLEXPORT extern "C" __declspec(dllexport)

__global__ void check_point_in_tet(
    float* point_coordinates_buf,
    float* point_velocities_buf,
    float* friction_velocity_buf,
    int* influence_count_buf,
    int* tet_vertex_IDs_buf,
    int* point_counts,
    int* tet_counts,
    int* is_colliding,
    int pid,
    int mesh_target,
    int mesh_collider
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < tet_counts[mesh_collider])
    {
        int* tet_vert_ids = &tet_vertex_IDs_buf[(idx + get_mesh_offset(tet_counts, mesh_collider)) * 4];
        float* tpos[4];
        for(int i = 0; i < 4; i++) tpos[i] = &point_coordinates_buf[tet_vert_ids[i] * 3];
        float* tvel[4];
        for(int i = 0; i < 4; i++) tvel[i] = &point_velocities_buf[tet_vert_ids[i] * 3];

        if(is_point_in_tet(tpos[0], tpos[1], tpos[2], tpos[3], &point_coordinates_buf[pid * 3], 1, 0.0))
        {
            atomicOr(&is_colliding[pid], 1);
            float fric_vel[3] = {0.0, 0.0, 0.0};
            for(int i = 0; i < 4; i++) 
            {
                add_vectors(fric_vel, tvel[i], fric_vel);
            }
            scale_vector(fric_vel, fric_vel, 1.0/4.0);
            add_vectors(fric_vel, &friction_velocity_buf[pid * 3], &friction_velocity_buf[pid * 3]);
        }
    }
}

__global__ void check_points_in_tet(
    float* point_coordinates_buf,
    float* point_velocities_buf,
    float* friction_velocity_buf,
    int* influence_count_buf,
    int* tet_vertex_IDs_buf,
    int* point_counts,
    int* tet_counts,
    int* is_colliding,
    int mesh_target,
    int mesh_collider
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < point_counts[mesh_target])
    {
        int pid = idx + get_mesh_offset(point_counts, mesh_target);
        int blockSize = 256;
        int gridSize = (tet_counts[mesh_collider] + blockSize - 1) / blockSize;
        check_point_in_tet<<<gridSize, blockSize>>>(
            point_coordinates_buf,
            point_velocities_buf,
            friction_velocity_buf,
            influence_count_buf,
            tet_vertex_IDs_buf,
            point_counts,
            tet_counts,
            is_colliding,
            pid,
            mesh_target,
            mesh_collider
        );
    }
}

__global__ void accumulate_radii(
    int* edge_vertex_IDs,
    float* edge_lengths,
    int edge_count,
    int* influence_count,
    float* radii
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < edge_count)
    {
        int id0 = edge_vertex_IDs[idx * 2];
        int id1 = edge_vertex_IDs[idx * 2 + 1];
        atomicAdd(&radii[id0], edge_lengths[idx] / 2.0);
        atomicAdd(&radii[id1], edge_lengths[idx] / 2.0);
        atomicAdd(&influence_count[id0], 1);
        atomicAdd(&influence_count[id1], 1);
    }
}

__global__ void avg_radii(
    float* radii,
    int* influence_counts,
    int count
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < count)
    {
        radii[idx] = radii[idx] / influence_counts[idx];
    }
}

__global__ void check_point_on_mesh(
    float* point_coordinates_buf,
    float* point_velocities_buf,
    float* point_inv_masses_buf,
    float* friction_velocity_buf,
    float* total_contribution_buf,
    int* influence_count_buf,
    int* point_counts,
    int* is_colliding,
    int pid,
    int mesh_target,
    int mesh_cloth,
    float* point_radii,
    float* multipliers
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < point_counts[mesh_cloth])
    {
        int cpid = idx + get_mesh_offset(point_counts, mesh_cloth);
        float* p = &point_coordinates_buf[pid * 3]; 
        float* cp = &point_coordinates_buf[cpid * 3];
        float dist = sqrtf(sqrd_distance_between(p, cp));
        
        float w0 = point_inv_masses_buf[pid];
        float w1 = point_inv_masses_buf[cpid];
        float w = w0+w1;

        if(w != 0.0)
        {
            float correction_gradient[3] = {0.0, 0.0, 0.0};
            sub_vectors(p, cp, correction_gradient);

            float length = sqrtf(sqrd_length_of(correction_gradient));
            float rest_length = ((point_radii[pid] * multipliers[mesh_target]) + (point_radii[cpid] * multipliers[mesh_cloth]));

            if((length != 0.0) && (length <= rest_length))
            {
                scale_vector(correction_gradient, correction_gradient, 1 / length); //normalize gradient
                float error = length - rest_length;

                float correction_magnitude = 0.0;

                float alpha = 0;
                correction_magnitude = (-1.0 * error) / (w + alpha);

                float correction_vec0[3] = {0.0, 0.0, 0.0};
                float correction_vec1[3] = {0.0, 0.0, 0.0};

                scale_vector(correction_gradient, correction_vec0, correction_magnitude * w0);
                scale_vector(correction_gradient, correction_vec1, -1.0 * correction_magnitude * w1);

                atomicAdd(&total_contribution_buf[pid * 3], correction_vec0[0]);
                atomicAdd(&total_contribution_buf[pid * 3 + 1], correction_vec0[1]);
                atomicAdd(&total_contribution_buf[pid * 3 + 2], correction_vec0[2]);

                atomicAdd(&total_contribution_buf[cpid * 3], correction_vec1[0]);
                atomicAdd(&total_contribution_buf[cpid * 3 + 1], correction_vec1[1]);
                atomicAdd(&total_contribution_buf[cpid * 3 + 2], correction_vec1[2]);

                atomicAdd(&influence_count_buf[pid], 1);
                atomicAdd(&influence_count_buf[cpid], 1);
            }
        }
    }
}

__global__ void check_mesh_against_mesh(
    float* point_coordinates_buf,
    float* point_velocities_buf,
    float* point_inv_masses_buf,
    float* friction_velocity_buf,
    float* total_contribution_buf,
    int* influence_count_buf,
    int* point_counts,
    int* is_colliding,
    int mesh_target,
    int mesh_cloth,
    float* point_radii,
    float* multipliers
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < point_counts[mesh_target])
    {
        int pid = idx + get_mesh_offset(point_counts, mesh_target);
        int blockSize = 256;
        int gridSize = (point_counts[mesh_cloth] + blockSize - 1) / blockSize;
        check_point_on_mesh<<<gridSize, blockSize>>>(
            point_coordinates_buf,
            point_velocities_buf,
            point_inv_masses_buf,
            friction_velocity_buf,
            total_contribution_buf,
            influence_count_buf,
            point_counts,
            is_colliding,
            pid,
            mesh_target,
            mesh_cloth,
            point_radii,
            multipliers
        );
    }
}

void solve_mesh_cloth(
    float* point_coordinates_buf,
    float* point_velocities_buf,
    float* point_inv_masses_buf,
    float* friction_velocity_buf,
    float* total_contribution_buf,
    int* influence_count_buf,
    int* point_counts,
    int* point_counts_buf,
    int* is_colliding,
    int mesh_target,
    int mesh_cloth,
    float* point_radii,
    float* multipliers
)
{
    int blockSize = 256;
    int gridSize = (point_counts[mesh_target] + blockSize - 1) / blockSize;
    check_mesh_against_mesh<<<gridSize, blockSize>>>(
        point_coordinates_buf,
        point_velocities_buf,
        point_inv_masses_buf,
        friction_velocity_buf,
        total_contribution_buf,
        influence_count_buf,
        point_counts_buf,
        is_colliding,
        mesh_target,
        mesh_cloth,
        point_radii,
        multipliers
    );
}

__global__ void generate_corrections(
    float* point_coordinates,
    float* correction_vectors,
    int* edge_vertex_IDs,
    int* point_counts,
    int* edge_counts,
    int* in_out,
    int target_mesh_id
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int mesh_point_offset = get_mesh_offset(point_counts, target_mesh_id);
    int mesh_edge_offset = get_mesh_offset(edge_counts, target_mesh_id);
    if(idx < edge_counts[target_mesh_id])
    {
        int id0 = edge_vertex_IDs[(mesh_edge_offset + idx) * 2];
        int id1 = edge_vertex_IDs[(mesh_edge_offset + idx) * 2 + 1];
        if(in_out[id0] == 1 && in_out[id1] == 0)
        { 
            float correction_vector[3] = {0.0, 0.0, 0.0};
            sub_vectors(&point_coordinates[id1 * 3], &point_coordinates[id0 * 3], correction_vector);
            atomicAdd(&correction_vectors[id0 * 3], correction_vector[0]);
            atomicAdd(&correction_vectors[id0 * 3 + 1], correction_vector[1]);
            atomicAdd(&correction_vectors[id0 * 3 + 2], correction_vector[2]);
        }
        else if(in_out[id0] == 0 && in_out[id1] == 1)
        {
            float correction_vector[3] = {0.0, 0.0, 0.0};
            sub_vectors(&point_coordinates[id0 * 3], &point_coordinates[id1 * 3], correction_vector);
            atomicAdd(&correction_vectors[id1 * 3], correction_vector[0]);
            atomicAdd(&correction_vectors[id1 * 3 + 1], correction_vector[1]);
            atomicAdd(&correction_vectors[id1 * 3 + 2], correction_vector[2]);
        }
    }
}

__global__ void normalize_vector_array(
    float* array,
    int vector_count
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < vector_count)
    {
        normalize_vector_to(&array[idx * 3], &array[idx * 3]);
    }
}

__global__ void project_to_surface_tri(
    float* point_coordinates,
    float* correction_vectors,
    float* correction_output,
    int* point_counts,
    int* surface_tris,
    int* surface_tri_counts,
    int target_id,
    int collider_id,
    int point_id,
    int* in_out
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < surface_tri_counts[collider_id])
    {
        if((correction_vectors[point_id * 3] > 0) || (correction_vectors[point_id * 3 + 1] > 0) || (correction_vectors[point_id * 3 + 2] > 0))
        {
            int triid = idx + get_mesh_offset(surface_tri_counts, collider_id);
            int* triv = &surface_tris[triid * 3];
            float* tripos[3];
            for(int i = 0; i < 3; i++) tripos[i] = &point_coordinates[triv[i] * 3];
            float intersection[3];
            if(triangle_ray_tracer(tripos[0], tripos[1], tripos[2], &point_coordinates[point_id * 3], &correction_vectors[point_id * 3], intersection))
            {   
                sub_vectors(intersection, &point_coordinates[point_id * 3], &correction_output[point_id * 3]);
                float correction_length = sqrt(sqrd_length_of(&correction_output[point_id * 3]));
                float correction_limit = 0.02;
                if(correction_length > correction_limit)
                {
                    scale_vector(&correction_output[point_id * 3], &correction_output[point_id * 3], correction_limit / correction_length);
                }
                atomicAnd(&in_out[point_id], 0);
            }
        }
    }
}

__global__ void project_point_to_tet(
    float* point_coordinates,
    float* correction_vectors,
    float* correction_output,
    int* surface_tris,
    int* surface_tri_counts,
    int* in_out,
    int* point_counts,
    int target_id,
    int collider_id
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < point_counts[target_id])
    {
        int pid = idx + get_mesh_offset(point_counts, target_id);
        int blockSize = 256;
        int gridSize = (surface_tri_counts[collider_id] + blockSize - 1) / blockSize;
        project_to_surface_tri<<<gridSize, blockSize>>>(
            point_coordinates,
            correction_vectors,
            correction_output,
            point_counts,
            surface_tris,
            surface_tri_counts,
            target_id,
            collider_id,
            pid,
            in_out
        );
    }
}

void solve_tet_tet_collisions_single(
    float* point_coordinates_buf,
    float* point_velocities_buf,
    float* total_contribution_buf,
    float* friction_velocity_buf,
    float* collision_correction_vector_buf,
    int* influence_count_buf,
    int* edge_vertex_IDs_buf,
    int* tet_vertex_IDs_buf,
    int* point_counts,
    int* point_counts_buf,
    int* edge_counts,
    int* edge_counts_buf,
    int* tet_counts,
    int* tet_counts_buf,
    int* surface_triangles,
    int* surface_triangles_buf,
    int* surface_tri_counts,
    int* surface_tri_counts_buf,
    int* is_colliding_buf,
    int mesh_target,
    int mesh_collider
)
{
    int blockSize = 256;
    int gridSize = (point_counts[mesh_target] + blockSize - 1) / blockSize;
    check_points_in_tet<<<gridSize, blockSize>>>(
        point_coordinates_buf,
        point_velocities_buf,
        friction_velocity_buf,
        influence_count_buf,
        tet_vertex_IDs_buf,
        point_counts_buf,
        tet_counts_buf,
        is_colliding_buf,
        mesh_target,
        mesh_collider
    );
    int iter = 0;
    while((iter < 10))
    {
        iter++;
        int blockSize = 256;
        int gridSize = (edge_counts[mesh_target] + blockSize - 1) / blockSize;
        generate_corrections<<<gridSize, blockSize>>>(
            point_coordinates_buf,
            collision_correction_vector_buf,
            edge_vertex_IDs_buf,
            point_counts_buf,
            edge_counts_buf,
            is_colliding_buf,
            mesh_target
        );
        cudaDeviceSynchronize();
        gridSize = (point_counts[mesh_target] + blockSize - 1) / blockSize;
        normalize_vector_array<<<gridSize, blockSize>>>(collision_correction_vector_buf, point_counts[mesh_target]);
        project_point_to_tet<<<gridSize, blockSize>>>(
            point_coordinates_buf,
            collision_correction_vector_buf,
            total_contribution_buf,
            surface_triangles_buf,
            surface_tri_counts_buf,
            is_colliding_buf,
            point_counts_buf,
            mesh_target,
            mesh_collider
        );
    }
}

__global__ void add_vector_array(
    float* array_1,
    float* array_2,
    float* array_out,
    int vector_count
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < vector_count)
    {
        add_vectors(&array_1[idx * 3], &array_2[idx * 3], &array_out[idx * 3]);
    }
}

void solve_tet_tet_collisions_double(
    float* point_coordinates_buf,
    float* point_velocities_buf,
    float* total_contribution_buf,
    float* friction_velocity_buf,
    float* collision_correction_vector_buf,
    int* influence_count_buf,
    int* edge_vertex_IDs_buf,
    int* tet_vertex_IDs_buf,
    int* point_counts,
    int* point_counts_buf,
    int* edge_counts,
    int* edge_counts_buf,
    int* tet_counts,
    int* tet_counts_buf,
    int* surface_triangles,
    int* surface_triangles_buf,
    int* surface_tri_counts,
    int* surface_tri_counts_buf,
    int* is_colliding_buf,
    int point_count,
    int mesh_1,
    int mesh_2
)
{
    int blockSize = 256;
    int gridSize = (point_counts[mesh_1] + blockSize - 1) / blockSize;
    check_points_in_tet<<<gridSize, blockSize>>>(
        point_coordinates_buf,
        point_velocities_buf,
        friction_velocity_buf,
        influence_count_buf,
        tet_vertex_IDs_buf,
        point_counts_buf,
        tet_counts_buf,
        is_colliding_buf,
        mesh_1,
        mesh_2
    );
    gridSize = (point_counts[mesh_2] + blockSize - 1) / blockSize;
    check_points_in_tet<<<gridSize, blockSize>>>(
        point_coordinates_buf,
        point_velocities_buf,
        friction_velocity_buf,
        influence_count_buf,
        tet_vertex_IDs_buf,
        point_counts_buf,
        tet_counts_buf,
        is_colliding_buf,
        mesh_2,
        mesh_1
    );
    int iter = 0;
    while((iter < 5))
    {
        iter++;
        gridSize = (edge_counts[mesh_1] + blockSize - 1) / blockSize;
        generate_corrections<<<gridSize, blockSize>>>(
            point_coordinates_buf,
            collision_correction_vector_buf,
            edge_vertex_IDs_buf,
            point_counts_buf,
            edge_counts_buf,
            is_colliding_buf,
            mesh_1
        );
        gridSize = (edge_counts[mesh_2] + blockSize - 1) / blockSize;
        generate_corrections<<<gridSize, blockSize>>>(
            point_coordinates_buf,
            collision_correction_vector_buf,
            edge_vertex_IDs_buf,
            point_counts_buf,
            edge_counts_buf,
            is_colliding_buf,
            mesh_2
        );
        gridSize = (point_count + blockSize - 1) / blockSize;
        normalize_vector_array<<<gridSize, blockSize>>>(collision_correction_vector_buf, point_count);
        gridSize = (point_counts[mesh_1] + blockSize - 1) / blockSize;
        project_point_to_tet<<<gridSize, blockSize>>>(
            point_coordinates_buf,
            collision_correction_vector_buf,
            total_contribution_buf,
            surface_triangles_buf,
            surface_tri_counts_buf,
            is_colliding_buf,
            point_counts_buf,
            mesh_1,
            mesh_2
        );
        gridSize = (point_counts[mesh_2] + blockSize - 1) / blockSize;
        project_point_to_tet<<<gridSize, blockSize>>>(
            point_coordinates_buf,
            collision_correction_vector_buf,
            total_contribution_buf,
            surface_triangles_buf,
            surface_tri_counts_buf,
            is_colliding_buf,
            point_counts_buf,
            mesh_2,
            mesh_1
        );
        gridSize = (point_count + blockSize - 1) / blockSize;
        add_vector_array<<<gridSize, blockSize>>>(point_coordinates_buf, total_contribution_buf, point_coordinates_buf, point_count);
        cudaMemset(total_contribution_buf, 0, 3 * point_count * sizeof(float));
        cudaMemset(collision_correction_vector_buf, 0, 3 * point_count * sizeof(float));
        cudaMemset(friction_velocity_buf, 0, 3 * point_count * sizeof(float));
    }
    cudaMemset(is_colliding_buf, 0, point_count * sizeof(int));
}

#endif