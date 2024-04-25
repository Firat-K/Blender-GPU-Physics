#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <random>
#include <cassert>
#include <cuda_runtime.h>
#include <set>
#include <unordered_set>
#include "vecmath.cuh"
#include "integrators.cuh"
#include "constraints.cuh"
#include "solver.h"
#include "collisions_narrow.cuh"
#include "collisions_broad.cuh"
#define DLLEXPORT extern "C" __declspec(dllexport)
#define PHYSICS_ON 1
#define COLLISIONS_ON 1
//int idx = threadIdx.x + blockIdx.x * blockDim.x;
//int blockSize = 256;
//int gridSize = 0;
//gridSize = (point_count + blockSize - 1) / blockSize;


float* point_coordinates_buf;
float* point_prev_coordinates_buf;
float* point_velocities_buf;
float* point_inv_masses_buf;

float* edge_rest_lengths_buf;
float* edge_tension_compliances_buf;
float* edge_compression_compliances_buf;
float* tet_rest_volumes_buf;
float* vol_compliances_buf;
float* bending_edge_rest_lengths_buf;
float* bending_compliances_buf;
float* edge_dampings_buf;
float* vol_dampings_buf;
float* bending_dampings_buf;
float* gravity_buf;
float* friction_velocity_buf;

int* edge_vertex_IDs_buf;
int* tet_vertex_IDs_buf;
int* tri_vertex_IDs_buf;
int* influence_count_buf;
int* vel_influence_count_buf;
int* point_counts_buf;
int* edge_counts_buf;
int* tet_counts_buf;
int* tri_counts_buf;
int* bending_edge_counts_buf;
int* point_in_out_buf;
int* surface_triangles_buf;
int* surface_tri_counts_buf;
int* bending_edge_vertex_IDs_buf;
int* is_colliding_buf;
float* total_contribution_buf;
float* collision_correction_vector_buf;
int* is_pinned_buf;
float* pin_coordinates_buf;

float* vertex_collider_radii_buf;
float* vertex_collider_radius_multiplier_buf;

void zero_contribs(int point_count)
{
    cudaMemset(total_contribution_buf, 0, 3 * point_count * sizeof(float));
    cudaMemset(collision_correction_vector_buf, 0, 3 * point_count * sizeof(float));
    cudaMemset(friction_velocity_buf, 0, 3 * point_count * sizeof(float));
    cudaMemset(influence_count_buf, 0, point_count * sizeof(int));
    cudaMemset(is_colliding_buf, 0, point_count * sizeof(int));
}

void apply_correction_buffer(float* contrib, int point_count, float dt_frame)
{
    int blockSize = 256;
    int gridSize = 0;
    gridSize = (point_count + blockSize - 1) / blockSize;
    apply_corrections<<<gridSize, blockSize>>>(point_coordinates_buf, point_inv_masses_buf, contrib, influence_count_buf, point_count, dt_frame);
    zero_contribs(point_count);
}

DLLEXPORT void solve_frame(
    float* point_coordinates,
    float* point_prev_coordinates,
    float* point_velocities,
    float* point_inv_masses,
    float* edge_rest_lengths,
    float* edge_tension_compliances,
    float* edge_compression_compliances,
    float* tet_rest_volumes,
    float* vol_compliances,
    float* bending_edge_rest_lengths, 
    float* bending_compliances, 
    float* edge_dampings,
    float* vol_dampings,
    float* bending_dampings,
    float* gravity,
    int* point_counts,
    int* edge_counts,
    int* tet_counts,
    int* tri_counts,
    int* bending_edge_counts, 
    int* edge_vertex_IDs,
    int* tet_vertex_IDs,
    int* tri_vertex_IDs,
    int* surface_triangles,
    int* surface_tri_counts,
    int* bending_edge_vertex_IDs,
    int* mesh_types,
    float* pin_coordinates,
    int* is_pinned,
    float* vertex_collider_radius_multiplier,
    int surface_tri_count,
    int point_count,
    int edge_count,
    int tet_count,
    int tri_count,
    int mesh_count,
    int bending_edge_count,
    int frequency,
    int sim_res,
    float floor_friction,
    float macro_cell_size
)
{
    float dt_sub = 1.0 / (frequency * sim_res);
    float dt_frame = 1.0 / frequency;

    int blockSize = 256;
    int gridSize = 0;
    {
    cudaMalloc((void**)&total_contribution_buf, 3 * point_count * sizeof(float));
    cudaMalloc((void**)&point_coordinates_buf, 3 * point_count * sizeof(float));
    cudaMalloc((void**)&point_prev_coordinates_buf, 3 * point_count * sizeof(float));
    cudaMalloc((void**)&point_velocities_buf, 3 * point_count * sizeof(float));
    cudaMalloc((void**)&point_inv_masses_buf, point_count * sizeof(float));
    cudaMalloc((void**)&edge_rest_lengths_buf, edge_count * sizeof(float));
    cudaMalloc((void**)&edge_tension_compliances_buf, edge_count * sizeof(float));
    cudaMalloc((void**)&edge_compression_compliances_buf, edge_count * sizeof(float));
    cudaMalloc((void**)&tet_rest_volumes_buf, tet_count * sizeof(float));
    cudaMalloc((void**)&vol_compliances_buf, tet_count * sizeof(float));
    cudaMalloc((void**)&bending_edge_rest_lengths_buf, bending_edge_count * sizeof(float));
    cudaMalloc((void**)&bending_compliances_buf, bending_edge_count * sizeof(float));
    cudaMalloc((void**)&edge_dampings_buf, edge_count * sizeof(float));
    cudaMalloc((void**)&vol_dampings_buf, tet_count * sizeof(float));
    cudaMalloc((void**)&bending_dampings_buf, bending_edge_count * sizeof(float));
    cudaMalloc((void**)&gravity_buf, 3 * sizeof(float));
    cudaMalloc((void**)&edge_vertex_IDs_buf, 2 * edge_count * sizeof(int));
    cudaMalloc((void**)&tet_vertex_IDs_buf, 4 * tet_count * sizeof(int));
    cudaMalloc((void**)&tri_vertex_IDs_buf, 3 * tri_count * sizeof(int));
    cudaMalloc((void**)&point_counts_buf, mesh_count * sizeof(int));
    cudaMalloc((void**)&edge_counts_buf, mesh_count * sizeof(int));
    cudaMalloc((void**)&tet_counts_buf, mesh_count * sizeof(int));
    cudaMalloc((void**)&tri_counts_buf, mesh_count * sizeof(int));
    cudaMalloc((void**)&bending_edge_counts_buf, mesh_count * sizeof(int));
    cudaMalloc((void**)&point_in_out_buf, point_count * sizeof(int));
    cudaMalloc((void**)&influence_count_buf, point_count * sizeof(int));
    cudaMalloc((void**)&vel_influence_count_buf, point_count * sizeof(int));
    cudaMalloc((void**)&surface_triangles_buf, 3 * surface_tri_count * sizeof(int));
    cudaMalloc((void**)&surface_tri_counts_buf, mesh_count * sizeof(int));
    cudaMalloc((void**)&bending_edge_vertex_IDs_buf, 2 * bending_edge_count * sizeof(int));
    cudaMalloc((void**)&collision_correction_vector_buf, 3 * point_count * sizeof(float));
    cudaMalloc((void**)&is_colliding_buf, point_count * sizeof(int));
    cudaMalloc((void**)&friction_velocity_buf, 3 * point_count * sizeof(float));
    cudaMalloc((void**)&is_pinned_buf, point_count * sizeof(int));
    cudaMalloc((void**)&pin_coordinates_buf, 3 * point_count * sizeof(float));
    cudaMalloc((void**)&vertex_collider_radii_buf, point_count * sizeof(float));
    cudaMalloc((void**)&vertex_collider_radius_multiplier_buf, mesh_count * sizeof(float));
    }
    //printf("Malloc Done \n");
//------------------------------------------------------------------------------------------------------------------
    {
    cudaMemcpy(point_coordinates_buf, point_coordinates, 3 * point_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(point_prev_coordinates_buf, point_prev_coordinates, 3 * point_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(point_velocities_buf, point_velocities, 3 * point_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(point_inv_masses_buf, point_inv_masses, point_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_rest_lengths_buf, edge_rest_lengths, edge_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_tension_compliances_buf, edge_tension_compliances, edge_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_compression_compliances_buf, edge_compression_compliances, edge_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tet_rest_volumes_buf, tet_rest_volumes, tet_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vol_compliances_buf, vol_compliances, tet_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bending_edge_rest_lengths_buf, bending_edge_rest_lengths, bending_edge_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bending_compliances_buf, bending_compliances, bending_edge_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_dampings_buf, edge_dampings, edge_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vol_dampings_buf, vol_dampings, tet_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bending_dampings_buf, bending_dampings, bending_edge_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gravity_buf, gravity, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_vertex_IDs_buf, edge_vertex_IDs, 2 * edge_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tet_vertex_IDs_buf, tet_vertex_IDs, 4 * tet_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tri_vertex_IDs_buf, tri_vertex_IDs, 3 * tri_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(point_counts_buf, point_counts, mesh_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_counts_buf, edge_counts, mesh_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tet_counts_buf, tet_counts, mesh_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tri_counts_buf, tri_counts, mesh_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bending_edge_counts_buf, bending_edge_counts, mesh_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(point_in_out_buf, 0, point_count * sizeof(int));
    cudaMemcpy(surface_triangles_buf, surface_triangles, 3 * surface_tri_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(surface_tri_counts_buf, surface_tri_counts, mesh_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bending_edge_vertex_IDs_buf, bending_edge_vertex_IDs, 2 * bending_edge_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(total_contribution_buf, 0, 3 * point_count * sizeof(float));
    cudaMemset(collision_correction_vector_buf, 0, 3 * point_count * sizeof(float));
    cudaMemset(influence_count_buf, 0, point_count * sizeof(int));
    cudaMemset(is_colliding_buf, 0, point_count * sizeof(int));
    cudaMemset(friction_velocity_buf, 0, point_count * sizeof(float));
    cudaMemset(vertex_collider_radii_buf, 0, point_count * sizeof(float));
    cudaMemcpy(is_pinned_buf, is_pinned, point_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pin_coordinates_buf, pin_coordinates, 3 * point_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vertex_collider_radius_multiplier_buf, vertex_collider_radius_multiplier, mesh_count * sizeof(float), cudaMemcpyHostToDevice);
    }
    //-------------------------------------------------------------------------------------------------------
    
    for(int substep = 0; substep < sim_res; substep++)
    {
        #if PHYSICS_ON
        gridSize = (point_count + blockSize - 1) / blockSize;
        integrate_velocities<<<gridSize, blockSize>>>(point_coordinates_buf, point_prev_coordinates_buf, point_inv_masses_buf, point_velocities_buf, point_count, dt_sub);
        solve_pin_constraint<<<gridSize, blockSize>>>(point_coordinates_buf, pin_coordinates_buf, is_pinned_buf, point_count);

        cudaMemset(vertex_collider_radii_buf, 0, point_count * sizeof(float));
        cudaMemset(influence_count_buf, 0, point_count * sizeof(int));
        gridSize = (edge_count + blockSize - 1) / blockSize;
        accumulate_radii<<<gridSize, blockSize>>>(edge_vertex_IDs_buf, edge_rest_lengths_buf, edge_count, influence_count_buf, vertex_collider_radii_buf);
        gridSize = (point_count + blockSize - 1) / blockSize;
        avg_radii<<<gridSize, blockSize>>>(vertex_collider_radii_buf, influence_count_buf, point_count);
        cudaMemset(influence_count_buf, 0, point_count * sizeof(int));

        for(int i = 0; i < mesh_count; i++)
        {
            for(int j = 0; j < mesh_count; j++)
            {
                if(i == j)
                {

                }
                else
                {
                    if((mesh_types[i] == 1) && (mesh_types[j] == 1))
                    {
                        solve_mesh_cloth(
                            point_coordinates_buf,
                            point_velocities_buf,
                            point_inv_masses_buf,
                            friction_velocity_buf,
                            total_contribution_buf,
                            influence_count_buf,
                            point_counts,
                            point_counts_buf,
                            is_colliding_buf,
                            i,
                            j,
                            vertex_collider_radii_buf,
                            vertex_collider_radius_multiplier_buf
                        );
                        apply_correction_buffer(total_contribution_buf, point_count, dt_sub);
                        solve_tet_tet_collisions_double(
                            point_coordinates_buf,
                            point_velocities_buf,
                            total_contribution_buf,
                            friction_velocity_buf,
                            collision_correction_vector_buf,
                            influence_count_buf,
                            edge_vertex_IDs_buf,
                            tet_vertex_IDs_buf,
                            point_counts,
                            point_counts_buf,
                            edge_counts,
                            edge_counts_buf,
                            tet_counts,
                            tet_counts_buf,
                            surface_triangles,
                            surface_triangles_buf,
                            surface_tri_counts,
                            surface_tri_counts_buf,
                            is_colliding_buf,
                            point_count,
                            i,
                            j
                        );
                    }
                    else if (((mesh_types[i] == 1) && (mesh_types[j] == 2)) || ((mesh_types[i] == 2) && (mesh_types[j] == 1)))
                    {
                        solve_mesh_cloth(
                            point_coordinates_buf,
                            point_velocities_buf,
                            point_inv_masses_buf,
                            friction_velocity_buf,
                            total_contribution_buf,
                            influence_count_buf,
                            point_counts,
                            point_counts_buf,
                            is_colliding_buf,
                            i,
                            j,
                            vertex_collider_radii_buf,
                            vertex_collider_radius_multiplier_buf
                        );
                        apply_correction_buffer(total_contribution_buf, point_count, dt_sub);
                    }
                }
            }
        }

        gridSize = (edge_count + blockSize - 1) / blockSize;
        if(edge_count > 0) 
        {
            solve_edge_constraints<<<gridSize, blockSize>>>(point_coordinates_buf, point_inv_masses_buf, edge_rest_lengths_buf, edge_compression_compliances_buf, edge_tension_compliances_buf, total_contribution_buf, edge_vertex_IDs_buf, influence_count_buf, edge_count, dt_sub);
            apply_correction_buffer(total_contribution_buf, point_count, dt_sub);
        }

        gridSize = (bending_edge_count + blockSize - 1) / blockSize;
        if(bending_edge_count > 0)
        {
            solve_edge_constraints<<<gridSize, blockSize>>>(point_coordinates_buf, point_inv_masses_buf, bending_edge_rest_lengths_buf, bending_compliances_buf, bending_compliances_buf, total_contribution_buf, bending_edge_vertex_IDs_buf, influence_count_buf, bending_edge_count, dt_sub);
            apply_correction_buffer(total_contribution_buf, point_count, dt_sub);
        }

        gridSize = (tet_count + blockSize - 1) / blockSize;
        if(tet_count > 0) 
        {
            solve_volume_constraints<<<gridSize, blockSize>>>(point_coordinates_buf, point_inv_masses_buf, tet_rest_volumes_buf, vol_compliances_buf, total_contribution_buf, tet_vertex_IDs_buf, influence_count_buf, tet_count, dt_sub);
            apply_correction_buffer(total_contribution_buf, point_count, dt_sub);
        }

        gridSize = (point_count + blockSize - 1) / blockSize;
        solve_floor_constraint<<<gridSize, blockSize>>>(point_coordinates_buf, point_prev_coordinates_buf, point_count);

        gridSize = (point_count + blockSize - 1) / blockSize;
        update_velocities<<<gridSize, blockSize>>>(point_coordinates_buf, point_prev_coordinates_buf, point_inv_masses_buf, point_velocities_buf, gravity_buf, point_count, dt_sub);

        gridSize = (edge_count + blockSize - 1) / blockSize;
        if(edge_count > 0) 
        {
            apply_edge_damping<<<gridSize, blockSize>>>(point_coordinates_buf, point_velocities_buf, total_contribution_buf, edge_dampings_buf, edge_vertex_IDs_buf, influence_count_buf, dt_sub, edge_count);
            apply_correction_buffer(total_contribution_buf, point_count, dt_sub);

            gridSize = (point_count + blockSize - 1) / blockSize;
            update_velocities<<<gridSize, blockSize>>>(point_coordinates_buf, point_prev_coordinates_buf, point_inv_masses_buf, point_velocities_buf, gravity_buf, point_count, dt_sub);
        }

        gridSize = (tet_count + blockSize - 1) / blockSize;
        if(tet_count > 0) 
        {
            apply_vol_dampings<<<gridSize, blockSize>>>(point_coordinates_buf, point_velocities_buf, total_contribution_buf, vol_dampings_buf, tet_vertex_IDs_buf, influence_count_buf, dt_sub, tet_count);
            apply_correction_buffer(total_contribution_buf, point_count, dt_sub);

            gridSize = (point_count + blockSize - 1) / blockSize;
            update_velocities<<<gridSize, blockSize>>>(point_coordinates_buf, point_prev_coordinates_buf, point_inv_masses_buf, point_velocities_buf, gravity_buf, point_count, dt_sub);
        }
        #endif

        gridSize = (point_count + blockSize - 1) / blockSize;
        apply_floor_friction<<<gridSize, blockSize>>>(point_coordinates_buf, point_velocities_buf, floor_friction, point_count);
    }
    
    //-------------------------------------------------------------------------------------------------------
    cudaMemcpy(point_coordinates, point_coordinates_buf, 3 * point_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(point_prev_coordinates, point_prev_coordinates_buf, 3 * point_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(point_velocities, point_velocities_buf, 3 * point_count * sizeof(float), cudaMemcpyDeviceToHost); 
    
    {
    cudaFree(total_contribution_buf);
    cudaFree(point_coordinates_buf);
    cudaFree(point_prev_coordinates_buf);
    cudaFree(point_velocities_buf);
    cudaFree(point_inv_masses_buf);
    cudaFree(edge_rest_lengths_buf);
    cudaFree(edge_tension_compliances_buf);
    cudaFree(edge_compression_compliances_buf);
    cudaFree(tet_rest_volumes_buf);
    cudaFree(vol_compliances_buf);
    cudaFree(bending_edge_rest_lengths_buf);
    cudaFree(bending_compliances_buf);
    cudaFree(edge_dampings_buf);
    cudaFree(vol_dampings_buf);
    cudaFree(bending_dampings_buf);
    cudaFree(gravity_buf);
    cudaFree(edge_vertex_IDs_buf);
    cudaFree(tet_vertex_IDs_buf);
    cudaFree(tri_vertex_IDs_buf);
    cudaFree(point_counts_buf);
    cudaFree(edge_counts_buf);
    cudaFree(tet_counts_buf);
    cudaFree(tri_counts_buf);
    cudaFree(bending_edge_counts_buf);
    cudaFree(point_in_out_buf);
    cudaFree(influence_count_buf);
    cudaFree(vel_influence_count_buf);
    cudaFree(surface_triangles_buf);
    cudaFree(surface_tri_counts_buf);
    cudaFree(bending_edge_vertex_IDs_buf);
    cudaFree(is_colliding_buf);
    cudaFree(friction_velocity_buf);
    cudaFree(collision_correction_vector_buf);
    cudaFree(is_pinned_buf);
    cudaFree(pin_coordinates_buf);
    cudaFree(vertex_collider_radii_buf);
    cudaFree(vertex_collider_radius_multiplier_buf);
    }
}

int main()
{

}
