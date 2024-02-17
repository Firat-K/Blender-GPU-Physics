#ifndef CONSTRAINTS_CUH_
#define CONSTRAINTS_CUH_

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "vecmath.cuh"
#define DLLEXPORT extern "C" __declspec(dllexport)

__global__ void solve_edge_constraints(
    float* point_coordinates,
    float* point_inv_masses,
    float* edge_rest_lengths,
    float* edge_compression_compliances,
    float* edge_tension_compliances,
    float* contribution,
    int* edge_vertex_IDs,
    int* edge_constraint_colors,
    int edge_constraint_count,
    int target_color,
    float dt
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int e_idx = idx * 2;

    if((idx < edge_constraint_count))
    {
        if(edge_constraint_colors[idx] == target_color)
        {
            float alpha_compression = edge_compression_compliances[idx] / powf(dt, 2.0);
            float alpha_tension = edge_tension_compliances[idx] / powf(dt, 2.0);

            int id0 = edge_vertex_IDs[e_idx];
            int id1 = edge_vertex_IDs[e_idx + 1];

            float w0 = point_inv_masses[id0];
            float w1 = point_inv_masses[id1];
            float w = w0+w1;

            if(w != 0.0)
            {
                float correction_gradient[3] = {0.0, 0.0, 0.0};
                sub_vectors(&point_coordinates[id0 * 3], &point_coordinates[id1 * 3], correction_gradient);

                float length = sqrtf(sqrd_length_of(correction_gradient));
                float rest_length = edge_rest_lengths[idx];

                if(length != 0.0)
                {
                    scale_vector(correction_gradient, correction_gradient, 1 / length); //normalize gradient
                    float error = length - rest_length;

                    float correction_magnitude = 0.0;

                    float alpha = (length < rest_length) ? alpha_compression : alpha_tension;
                    correction_magnitude = (-1.0 * error) / (w + alpha);

                    float correction_vec0[3] = {0.0, 0.0, 0.0};
                    float correction_vec1[3] = {0.0, 0.0, 0.0};

                    scale_vector(correction_gradient, correction_vec0, correction_magnitude * w0);
                    scale_vector(correction_gradient, correction_vec1, -1.0 * correction_magnitude * w1);

                    add_vectors(&point_coordinates[id0 * 3], correction_vec0, &point_coordinates[id0 * 3]);
                    add_vectors(&point_coordinates[id1 * 3], correction_vec1, &point_coordinates[id1 * 3]);
                    
                    add_vectors(&contribution[id0 * 3], correction_vec0, &contribution[id0 * 3]);
                    add_vectors(&contribution[id1 * 3], correction_vec1, &contribution[id1 * 3]);
                }
            }
        }
    }
}

__global__ void solve_volume_constraints(
    float* point_coordinates,
    float* point_inv_masses,
    float* tet_rest_volumes,
    float* vol_constraint_compliances,
    float* contribution,
    int* tet_vertex_IDs,
    int* vol_constraint_colors,
    int vol_constraint_count,
    int target_color,
    float dt
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidx = idx * 4;
    float correction_gradients[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    float alpha = vol_constraint_compliances[idx] / powf(dt, 2.0);
    int tet_triangle_v_order[4][3] = {{1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}};
    if((idx < vol_constraint_count))
    {
        if(vol_constraint_colors[idx] == target_color)
        {
            float w = 0.0;

            for(int i = 0; i < 4; i++)
            {
                int id0 = tet_vertex_IDs[tidx + tet_triangle_v_order[i][0]];
                int id1 = tet_vertex_IDs[tidx + tet_triangle_v_order[i][1]];
                int id2 = tet_vertex_IDs[tidx + tet_triangle_v_order[i][2]];

                float vec_0[3] = {0.0, 0.0, 0.0};
                float vec_1[3] = {0.0, 0.0, 0.0};

                sub_vectors(&point_coordinates[id1 * 3], &point_coordinates[id0 * 3], vec_0);
                sub_vectors(&point_coordinates[id2 * 3], &point_coordinates[id0 * 3], vec_1);

                cross_product_of(vec_0, vec_1, correction_gradients[i]);
                scale_vector(correction_gradients[i], correction_gradients[i], 1.0/6.0);

                w += point_inv_masses[tet_vertex_IDs[tidx + i]] * sqrd_length_of(correction_gradients[i]);
            }

            if(w != 0.0)
            {
                float volume = tet_volume_from_buffer(tet_vertex_IDs, point_coordinates, idx);
                float rest_volume = tet_rest_volumes[idx];

                float error = volume - rest_volume;
                float correction_magnitude = (-1.0 * error) / (w + alpha);

                for(int i = 0; i < 4; i++)
                {
                    int id = tet_vertex_IDs[tidx + i];
                    float correction_vector[3] = {0.0, 0.0, 0.0};
                    scale_vector(correction_gradients[i], correction_vector, correction_magnitude * point_inv_masses[id]);
                    add_vectors(&point_coordinates[id * 3], correction_vector, &point_coordinates[id * 3]);
                    add_vectors(&contribution[id * 3], correction_vector, &contribution[id * 3]);
                }
            }
        }
    }
}

__global__ void solve_dampening_for_constraint(
    float* point_velocities,
    float* constraint_contribution,
    float damping_constant,
    int point_count,
    float dt
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < point_count)
    {
        float contribution[3] = {0.0, 0.0, 0.0};
        scale_vector(&constraint_contribution[idx * 3], contribution, 1.0 / dt);
        float vel_without_constraint[3] = {0.0, 0.0, 0.0};
        sub_vectors(&point_velocities[idx * 3], contribution, vel_without_constraint);
        scale_vector(contribution, contribution, 1.0 / damping_constant);
        add_vectors(vel_without_constraint, contribution, &point_velocities[idx * 3]);
    }
}

__global__ void solve_floor_constraint(
    float* point_coordinates,
    float* point_prev_coordinates,
    int point_count
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < point_count)
    {
        if(point_coordinates[idx * 3 + 2] < 0)
        {
            point_coordinates[idx * 3 + 2] = 0.0;
        }
    }
}


#endif /* CONSTRAINTS_CUH_ */
