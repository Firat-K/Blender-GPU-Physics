#ifndef INTEGRATORS_CUH_
#define INTEGRATORS_CUH_

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "vecmath.cuh"
#define DLLEXPORT extern "C" __declspec(dllexport)

__global__ void integrate_velocities(
    float* point_coordinates,
    float* point_prev_coodinates,
    float* point_inv_masses,
    float* point_velocities,
    int point_count,
    float dt
)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if((tidx < point_count))
    {
        if((point_inv_masses[tidx] != 0.0))
        {
            int pidx = tidx * 3;
            copy_vector_to(&point_coordinates[pidx], &point_prev_coodinates[pidx]);
            float vel_integrated[3] = {0.0, 0.0, 0.0};
            scale_vector(&point_velocities[pidx], vel_integrated, dt);
            add_vectors(&point_coordinates[pidx], vel_integrated, &point_coordinates[pidx]);
        }
    }
}

__global__ void apply_corrections(
    float* point_coordinates,
    float* point_inv_masses,
    float* total_contribution,
    int* influence_count,
    int point_count,
    float dt
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < point_count)
    {
        if(point_inv_masses[idx] != 0.0)
        {
            if(influence_count[idx] != 0)
            {
                int pidx = idx * 3;
                float correction[3]= {0.0, 0.0, 0.0};
                scale_vector(&total_contribution[pidx], correction, 1.0/influence_count[idx]);
                add_vectors(&point_coordinates[pidx], correction, &point_coordinates[pidx]);
            }
        }
    }
}

__global__ void apply_corrections_no_inf(
    float* point_coordinates,
    float* point_inv_masses,
    float* total_contribution,
    int point_count
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < point_count)
    {
        if(point_inv_masses[idx] != 0.0)
        {
            int pidx = idx * 3;
            add_vectors(&point_coordinates[pidx], &total_contribution[pidx], &point_coordinates[pidx]);
        }
    }
}


__global__ void update_velocities(
    float* point_coordinates,
    float* point_prev_coodinates,
    float* point_inv_masses,
    float* point_velocities,
    float* ext_forces,
    int point_count,
    float dt
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < point_count)
    {
        if(point_inv_masses[idx] != 0.0)
        {
            int pidx = idx * 3;
            float displacement[3] = {0.0, 0.0, 0.0};
            sub_vectors(&point_coordinates[pidx], &point_prev_coodinates[pidx], displacement);
            scale_vector(displacement, &point_velocities[pidx], 1.0 / dt);
            float ext_forces_dt[3] = {0.0, 0.0, 0.0};
            scale_vector(ext_forces, ext_forces_dt, dt);
            add_vectors(&point_velocities[pidx], ext_forces_dt, &point_velocities[pidx]);
        }
    }
}

__global__ void apply_floor_friction(
    float* point_coordinates,
    float* point_velocities,
    float friction,
    int point_count
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < point_count)
    {
        if(abs(point_coordinates[idx * 3 + 2]) < 0.001)
        {
            float vec_x[3] = {1.0, 0.0, 0.0};
            float vec_y[3] = {0.0, 1.0, 0.0};
            float* vec_v = &point_velocities[3 * idx];

            float dot_x_v = dot_product_of(vec_x, vec_v);
            float dot_y_v = dot_product_of(vec_y, vec_v);

            scale_vector(vec_x, vec_x, dot_x_v);
            scale_vector(vec_y, vec_y, dot_y_v);
            float floor_component[3] = {0.0, 0.0, 0.0};
            add_vectors(vec_x, vec_y, floor_component);
            float dissected_velocity[3] = {0.0, 0.0, 0.0};
            sub_vectors(vec_v, floor_component, dissected_velocity);
            scale_vector(floor_component, floor_component, 1/friction);
            add_vectors(floor_component, dissected_velocity, vec_v);
        }
    }
}

__global__ void apply_damping(
    float* point_coordinates,
    float* point_velocities,
    float* total_contribution,
    float* dampings,
    int* elem_vertex_IDs,
    int* influence_count,
    float dt,
    int elem_size,
    int elem_count
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < elem_count)
    {
        float vel_corrections[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        float avg_vel[3] = {0.0, 0.0, 0.0};
        int elem_vertex_ids[4] = {0, 0, 0, 0};
        for(int i = 0; i < elem_size; i++)
        {
            elem_vertex_ids[i] = elem_vertex_IDs[elem_size * idx + i];
        } 
        for(int i = 0; i < elem_size; i++)
        {
            add_vectors(avg_vel, &point_velocities[elem_vertex_ids[i] * 3], avg_vel);
        }       
        scale_vector(avg_vel, avg_vel, 1.0/elem_size);
        for(int i = 0; i < elem_size; i++)
        {
            sub_vectors(avg_vel, &point_velocities[elem_vertex_ids[i] * 3], &vel_corrections[i * 3]);
        }   
        for(int i = 0; i < elem_size; i++)
        {
            scale_vector(&vel_corrections[i*3], &vel_corrections[i*3], dt * dampings[idx]);
            atomicAdd(&total_contribution[elem_vertex_ids[i] * 3], vel_corrections[i*3]);
            atomicAdd(&total_contribution[elem_vertex_ids[i] * 3 + 1], vel_corrections[i*3 + 1]);
            atomicAdd(&total_contribution[elem_vertex_ids[i] * 3 + 2], vel_corrections[i*3 + 2]);
            if(sqrd_length_of(&vel_corrections[i * 3]) != 0) atomicAdd(&influence_count[elem_vertex_ids[i]], 1);
        }
    }
}

__global__ void apply_edge_damping(
    float* point_coordinates,
    float* point_velocities,
    float* total_contribution,
    float* dampings,
    int* edge_vertex_IDs,
    int* influence_count,
    float dt,
    int edge_count
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < edge_count)
    {
        int id0 = edge_vertex_IDs[idx * 2];
        int id1 = edge_vertex_IDs[idx * 2 + 1];
        float* p0 = &point_coordinates[id0 * 3];
        float* p1 = &point_coordinates[id1 * 3];
        float* vel0 = &point_velocities[id0 * 3];
        float* vel1 = &point_velocities[id1 * 3];
        float grad01[3] = {0.0, 0.0, 0.0};
        float grad10[3] = {0.0, 0.0, 0.0};
        sub_vectors(p1, p0 , grad01);
        sub_vectors(p0, p1, grad10);
        normalize_vector_to(grad01, grad01);
        normalize_vector_to(grad10, grad10);

        float comp01[3] = {0.0, 0.0, 0.0};
        float comp10[3] = {0.0, 0.0, 0.0};
        scale_vector(grad01, comp01, dot_product_of(vel0, grad01));
        scale_vector(grad10, comp10, dot_product_of(vel1, grad10));

        float comp_avg[3] = {0.0, 0.0, 0.0};
        add_vectors(comp_avg, comp01, comp_avg);
        add_vectors(comp_avg, comp10, comp_avg);
        scale_vector(comp_avg, comp_avg, 0.5);

        float tvel0[3] = {0.0, 0.0, 0.0};
        float tvel1[3] = {0.0, 0.0, 0.0};
        sub_vectors(vel0, comp01, tvel0);
        sub_vectors(vel1, comp10, tvel1);

        add_vectors(tvel0, comp_avg, tvel0);
        add_vectors(tvel1, comp_avg, tvel1);

        float corr0[3] = {0.0, 0.0, 0.0};
        float corr1[3] = {0.0, 0.0, 0.0};
        sub_vectors(tvel0, vel0, corr0);
        sub_vectors(tvel1, vel1, corr1);

        scale_vector(corr0, corr0, dt * dampings[idx]);
        scale_vector(corr1, corr1, dt * dampings[idx]);

        atomicAdd(&total_contribution[id0 * 3], corr0[0]);
        atomicAdd(&total_contribution[id0 * 3 + 1], corr0[1]);
        atomicAdd(&total_contribution[id0 * 3 + 2], corr0[2]);
        atomicAdd(&influence_count[id0], 1);

        atomicAdd(&total_contribution[id1 * 3], corr1[0]);
        atomicAdd(&total_contribution[id1 * 3 + 1], corr1[1]);
        atomicAdd(&total_contribution[id1 * 3 + 2], corr1[2]);
        atomicAdd(&influence_count[id1], 1);
    }
}

__global__ void apply_vol_dampings(
    float* point_coordinates,
    float* point_velocities,
    float* total_contribution,
    float* dampings,
    int* tet_vertex_IDs,
    int* influence_count,
    float dt,
    int tet_count
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < tet_count)
    {
        int tidx = idx * 4;
        int tet_triangle_v_order[4][3] = {{1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}};
        float correction_gradients[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

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
            //scale_vector(correction_gradients[i], correction_gradients[i], 1.0/6.0);
            normalize_vector_to(correction_gradients[i], correction_gradients[i]);
        }
        
        int ids[4] = {0,0,0,0};
        for(int i = 0; i < 4; i++)
        {
            ids[i] = tet_vertex_IDs[tidx + i];
        }

        float components[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        float component_avg[3] = {0.0, 0.0, 0.0};
        for(int i = 0; i < 4; i++)
        {
            scale_vector(correction_gradients[i], components[i], dot_product_of(&point_velocities[ids[i] * 3], correction_gradients[i]));
            add_vectors(component_avg, correction_gradients[i], component_avg);
        }
        scale_vector(component_avg, component_avg, 1.0/4.0);

        float target_vels[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        for(int i = 0; i < 4; i++)
        {
            sub_vectors(&point_velocities[ids[i] * 3], components[i], target_vels[i]);
            add_vectors(target_vels[i], component_avg, target_vels[i]);
        }

        float corrections[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        for(int i = 0; i < 4; i++)
        {
            sub_vectors(target_vels[i], &point_velocities[ids[i] * 3], corrections[i]);
            scale_vector(corrections[i], corrections[i], dt * dampings[idx]);

            atomicAdd(&total_contribution[ids[i] * 3], corrections[i][0]);
            atomicAdd(&total_contribution[ids[i] * 3 + 1], corrections[i][1]);
            atomicAdd(&total_contribution[ids[i] * 3 + 2], corrections[i][2]);
            atomicAdd(&influence_count[ids[i]], 1);
        }
    }
}
#endif /* INTEGRATORS_CUH_ */