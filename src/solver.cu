#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <random>
#include <cassert>
#include <cuda_runtime.h>
#include "vecmath.cuh"
#include "integrators.cuh"
#include "constraints.cuh"
#include "collisions.cuh"
#define DLLEXPORT extern "C" __declspec(dllexport)
#define SOFTBODY 1
#define PHYSICS_ON 1
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
float* edge_constraint_contribution_buf;

float* tet_rest_volumes_buf;
float* vol_compliances_buf;
float* vol_constraint_contribution_buf;

float* gravity_buf;

int* edge_constraint_colors_buf;
int* edge_vertex_IDs_buf;

int* vol_constraint_colors_buf;
int* tet_vertex_IDs_buf;

int* point_counts_buf;
int* edge_counts_buf;
int* tet_counts_buf;

int* point_in_out_buf;
int* surface_triangles_buf;
int* surface_tri_counts_buf;

float* collision_correction_buf;

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
    float* gravity,
    int* point_counts,
    int* edge_counts,
    int* tet_counts,
    int* edge_constraint_colors,
    int* edge_vertex_IDs,
    int* vol_constraint_colors,
    int* tet_vertex_IDs,
    int* surface_triangles,
    int* surface_tri_counts,
    int surface_tri_count,
    int point_count,
    int edge_count,
    int tet_count,
    int mesh_count,
    int edge_constraint_color_count,
    int tet_constraint_color_count,
    int frequency,
    int sim_res
)
{
    float dt = 1.0 / (frequency * sim_res);

    int blockSize = 256;
    int gridSize = 0;

    cudaMalloc((void**)&point_coordinates_buf, 3 * point_count * sizeof(float));
    cudaMalloc((void**)&point_prev_coordinates_buf, 3 * point_count * sizeof(float));
    cudaMalloc((void**)&point_velocities_buf, 3 * point_count * sizeof(float));
    cudaMalloc((void**)&point_inv_masses_buf, point_count * sizeof(float));

    cudaMalloc((void**)&edge_rest_lengths_buf, edge_count * sizeof(float));
    cudaMalloc((void**)&edge_tension_compliances_buf, edge_count * sizeof(float));
    cudaMalloc((void**)&edge_compression_compliances_buf, edge_count * sizeof(float));
    cudaMalloc((void**)&edge_constraint_contribution_buf, 3 * point_count * sizeof(float));

    cudaMalloc((void**)&tet_rest_volumes_buf, tet_count * sizeof(float));
    cudaMalloc((void**)&vol_compliances_buf, tet_count * sizeof(float));
    cudaMalloc((void**)&vol_constraint_contribution_buf, 3 * point_count * sizeof(float));

    cudaMalloc((void**)&gravity_buf, 3 * sizeof(float));

    cudaMalloc((void**)&edge_constraint_colors_buf, edge_count * sizeof(int));
    cudaMalloc((void**)&edge_vertex_IDs_buf, 2 * edge_count * sizeof(int));

    cudaMalloc((void**)&vol_constraint_colors_buf, tet_count * sizeof(int));
    cudaMalloc((void**)&tet_vertex_IDs_buf, 4 * tet_count * sizeof(int));

    cudaMalloc((void**)&point_counts_buf, mesh_count * sizeof(int));
    cudaMalloc((void**)&edge_counts_buf, mesh_count * sizeof(int));
    cudaMalloc((void**)&tet_counts_buf, mesh_count * sizeof(int));

    cudaMalloc((void**)&point_in_out_buf, point_count * sizeof(int));
    cudaMalloc((void**)&surface_triangles_buf, 3 * surface_tri_count * sizeof(int));
    cudaMalloc((void**)&surface_tri_counts_buf, mesh_count * sizeof(int));

    cudaMalloc((void**)&collision_correction_buf, 3 * point_count * sizeof(float));
//------------------------------------------------------------------------------------------------------------------
    cudaMemcpy(point_coordinates_buf, point_coordinates, 3 * point_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(point_prev_coordinates_buf, point_prev_coordinates, 3 * point_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(point_velocities_buf, point_velocities, 3 * point_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(point_inv_masses_buf, point_inv_masses, point_count * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(edge_rest_lengths_buf, edge_rest_lengths, edge_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_tension_compliances_buf, edge_tension_compliances, edge_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_compression_compliances_buf, edge_compression_compliances, edge_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(edge_constraint_contribution_buf, 0, 3 * point_count * sizeof(float));

    cudaMemcpy(tet_rest_volumes_buf, tet_rest_volumes, tet_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vol_compliances_buf, vol_compliances, tet_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(vol_constraint_contribution_buf, 0, 3 * point_count * sizeof(float));

    cudaMemcpy(gravity_buf, gravity, 3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(edge_constraint_colors_buf, edge_constraint_colors, edge_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_vertex_IDs_buf, edge_vertex_IDs, 2 * edge_count * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(vol_constraint_colors_buf, vol_constraint_colors, tet_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tet_vertex_IDs_buf, tet_vertex_IDs, 4 * tet_count * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(point_counts_buf, point_counts, mesh_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_counts_buf, edge_counts, mesh_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tet_counts_buf, tet_counts, mesh_count * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(point_in_out_buf, 0, point_count * sizeof(int));
    cudaMemset(collision_correction_buf, 0, 3 * point_count * sizeof(float));
    cudaMemcpy(surface_triangles_buf, surface_triangles, 3 * surface_tri_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(surface_tri_counts_buf, surface_tri_counts, mesh_count * sizeof(int), cudaMemcpyHostToDevice);
    

    #if PHYSICS_ON
    for(int substep = 0; substep < sim_res; substep++)
    {
        cudaMemset(edge_constraint_contribution_buf, 0, 3 * point_count * sizeof(float));
        cudaMemset(vol_constraint_contribution_buf, 0, 3 * point_count * sizeof(float));

        std::vector<int> edge_color_random;
        for (int i = 1; i <= edge_constraint_color_count+1; ++i) {
            edge_color_random.push_back(i);
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(edge_color_random.begin(), edge_color_random.end(), gen);

        std::vector<int> vol_color_random;
        for (int i = 1; i <= tet_constraint_color_count+1; ++i) {
            vol_color_random.push_back(i);
        }
        std::shuffle(vol_color_random.begin(), vol_color_random.end(), gen);
        //----------------------------------------------------------------
        gridSize = (point_count + blockSize - 1) / blockSize;
        integrate_velocities<<<gridSize, blockSize>>>(
            point_coordinates_buf, 
            point_prev_coordinates_buf, 
            point_inv_masses_buf,
            point_velocities_buf,
            point_count,
            dt
            );
        
        solve_floor_constraint<<<gridSize, blockSize>>>(
            point_coordinates_buf, 
            point_prev_coordinates_buf, 
            point_count
            );
        
        //COLLISION START
        gridSize = (point_counts[0] + blockSize - 1) / blockSize;
        tet_tet_mark_inside<<<gridSize, blockSize>>>(
            point_coordinates_buf,
            tet_vertex_IDs_buf,
            point_counts_buf,
            tet_counts_buf,
            point_in_out_buf,
            1,
            0
        );
        for(int i = 0; i < 10; i++)
        {
            cudaMemset(collision_correction_buf, 0, 3 * point_count * sizeof(float));
            gridSize = (edge_counts[0] + blockSize - 1) / blockSize;
            set_tet_collision_correction_vectors<<<gridSize, blockSize>>>(
                point_coordinates_buf,
                point_counts_buf,
                point_in_out_buf,
                edge_vertex_IDs_buf,
                edge_counts_buf,
                collision_correction_buf,
                0
            );
            gridSize = (point_counts[0] + blockSize - 1) / blockSize;
            project_point_tet_mesh<<<gridSize, blockSize>>>(
                collision_correction_buf,
                point_coordinates_buf,
                point_counts_buf,
                surface_triangles_buf,
                surface_tri_counts_buf,
                point_in_out_buf,
                0,
                1
            );
        }
        //COLLISION END

        //COLLISION START
        gridSize = (point_counts[1] + blockSize - 1) / blockSize;
        tet_tet_mark_inside<<<gridSize, blockSize>>>(
            point_coordinates_buf,
            tet_vertex_IDs_buf,
            point_counts_buf,
            tet_counts_buf,
            point_in_out_buf,
            0,
            1
        );
        for(int i = 0; i < 10; i++)
        {
            cudaMemset(collision_correction_buf, 0, 3 * point_count * sizeof(float));
            gridSize = (edge_counts[1] + blockSize - 1) / blockSize;
            set_tet_collision_correction_vectors<<<gridSize, blockSize>>>(
                point_coordinates_buf,
                point_counts_buf,
                point_in_out_buf,
                edge_vertex_IDs_buf,
                edge_counts_buf,
                collision_correction_buf,
                1
            );
            gridSize = (point_counts[1] + blockSize - 1) / blockSize;
            project_point_tet_mesh<<<gridSize, blockSize>>>(
                collision_correction_buf,
                point_coordinates_buf,
                point_counts_buf,
                surface_triangles_buf,
                surface_tri_counts_buf,
                point_in_out_buf,
                1,
                0
            );
        }
        //COLLISION END

        for(int i = 0; i < edge_constraint_color_count; i++)
        {
            gridSize = (edge_count + blockSize - 1) / blockSize;
            solve_edge_constraints<<<gridSize, blockSize>>>(
                point_coordinates_buf,
                point_inv_masses_buf,
                edge_rest_lengths_buf,
                edge_compression_compliances_buf,
                edge_tension_compliances_buf,
                edge_constraint_contribution_buf,
                edge_vertex_IDs_buf,
                edge_constraint_colors_buf,
                edge_count,
                edge_color_random[i],
                dt
                );
        }
        
        for(int i = 0; i < tet_constraint_color_count; i++)
        {
            gridSize = (edge_count + blockSize - 1) / blockSize;
            solve_volume_constraints<<<gridSize, blockSize>>>(
                point_coordinates_buf,
                point_inv_masses_buf,
                tet_rest_volumes_buf,
                vol_compliances_buf,
                edge_constraint_contribution_buf,
                tet_vertex_IDs_buf,
                vol_constraint_colors_buf,
                tet_count,
                vol_color_random[i],
                dt
                );
        }

        gridSize = (point_count + blockSize - 1) / blockSize;
        update_velocities<<<gridSize, blockSize>>>(
            point_coordinates_buf,
            point_prev_coordinates_buf,
            point_inv_masses_buf,
            point_velocities_buf,
            gravity_buf,
            point_count,
            dt
            );
        //----------------------------------------------------------------
    }
    #endif

    cudaMemcpy(point_coordinates, point_coordinates_buf, 3 * point_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(point_prev_coordinates, point_prev_coordinates_buf, 3 * point_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(point_velocities, point_velocities_buf, 3 * point_count * sizeof(float), cudaMemcpyDeviceToHost);
    

    cudaFree(point_coordinates_buf);
    cudaFree(point_prev_coordinates_buf);
    cudaFree(point_velocities_buf);
    cudaFree(point_inv_masses_buf);

    cudaFree(edge_rest_lengths_buf);
    cudaFree(edge_tension_compliances_buf);
    cudaFree(edge_compression_compliances_buf);
    cudaFree(edge_constraint_contribution_buf);

    cudaFree(tet_rest_volumes_buf);
    cudaFree(vol_compliances_buf);
    cudaFree(vol_constraint_contribution_buf);

    cudaFree(gravity_buf);

    cudaFree(edge_constraint_colors_buf);
    cudaFree(edge_vertex_IDs_buf);

    cudaFree(vol_constraint_colors_buf);
    cudaFree(tet_vertex_IDs_buf);

    cudaFree(point_counts_buf);
    cudaFree(edge_counts_buf);
    cudaFree(tet_counts_buf);

    cudaFree(surface_tri_counts_buf);
    cudaFree(surface_triangles_buf);
}

int main()
{

}