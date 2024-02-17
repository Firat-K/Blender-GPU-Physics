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
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if(tidx < point_count)
    {
        if(point_inv_masses[tidx] != 0.0)
        {
            int pidx = tidx * 3;
            float displacement[3] = {0.0, 0.0, 0.0};
            sub_vectors(&point_coordinates[pidx], &point_prev_coodinates[pidx], displacement);
            scale_vector(displacement, &point_velocities[pidx], 1.0 / dt);
            float ext_forces_dt[3] = {0.0, 0.0, 0.0};
            scale_vector(ext_forces, ext_forces_dt, dt);
            add_vectors(&point_velocities[pidx], ext_forces_dt, &point_velocities[pidx]);
        }
    }
}

#endif /* INTEGRATORS_CUH_ */