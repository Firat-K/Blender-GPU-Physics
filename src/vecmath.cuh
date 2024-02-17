#ifndef VECMATH_CUH_
#define VECMATH_CUH_
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#define DLLEXPORT extern "C" __declspec(dllexport)
#define KERNEL_OK 1
#define KERNEL_ERROR 0
//int idx = threadIdx.x + blockIdx.x * blockDim.x;

__device__ void zero_vector(float* vec)
{
    vec[0] = 0.0;
    vec[1] = 0.0;
    vec[2] = 0.0;
}

__device__ void copy_vector_to(float* vec_in, float* vec_out)
{
    vec_out[0] = vec_in[0];
    vec_out[1] = vec_in[1];
    vec_out[2] = vec_in[2];
}

__device__ void add_vectors(float* vec_1, float* vec_2, float* vec_out)
{
    vec_out[0] = vec_1[0] + vec_2[0];
    vec_out[1] = vec_1[1] + vec_2[1];
    vec_out[2] = vec_1[2] + vec_2[2];
}

__device__ void sub_vectors(float* vec_1, float* vec_2, float* vec_out)
{
    vec_out[0] = vec_1[0] - vec_2[0];
    vec_out[1] = vec_1[1] - vec_2[1];
    vec_out[2] = vec_1[2] - vec_2[2];
}

__device__ void scale_vector(float* vec, float* vec_out, float scale)
{
    vec_out[0] = vec[0] * scale;
    vec_out[1] = vec[1] * scale;
    vec_out[2] = vec[2] * scale;
}

__device__ float sqrd_length_of(float* vec)
{
    return (powf(vec[0], 2.0) + powf(vec[1], 2.0) + powf(vec[2], 2.0));
}

__device__ float sqrd_distance_between(float* vec_1, float* vec_2)
{
    float disp_vec[3] = {0.0, 0.0, 0.0};
    sub_vectors(vec_1, vec_2, disp_vec);
    return sqrd_length_of(disp_vec);
}

__device__ void normalize_vector_to(float* vec, float* vec_out)
{
    scale_vector(vec, vec_out, 1.0/sqrtf(sqrd_length_of(vec)));
}

__device__ void average_vectors(float* vec_1, float* vec_2, float* vec_out)
{
    float temp[3] = {0.0, 0.0, 0.0};
    add_vectors(vec_1, vec_2, temp);
    scale_vector(temp, vec_out, 1.0/2.0);
}

__device__ float dot_product_of(float* vec_1, float* vec_2)
{
    return ((vec_1[0] * vec_2[0]) + (vec_1[1] * vec_2[1]) + (vec_1[2] * vec_2[2]));
}

__device__ void cross_product_of(float* vec_1, float* vec_2, float* vec_out)
{
    vec_out[0] = (vec_1[1] * vec_2[2]) - (vec_1[2] * vec_2[1]);
    vec_out[1] = (vec_1[2] * vec_2[0]) - (vec_1[0] * vec_2[2]);
    vec_out[2] = (vec_1[0] * vec_2[1]) - (vec_1[1] * vec_2[0]);
}

__device__ float tet_volume(float* vec_1, float* vec_2, float* vec_3)
{  
    float cross[3] = {0.0, 0.0, 0.0};
    cross_product_of(vec_1, vec_2, cross);
    return (dot_product_of(vec_3, cross) / 6.0);
}

__device__ float tet_volume_from_buffer(int* tet_vert_ids, float* v_positions, int n)
{
    int id0 = tet_vert_ids[4 * n];
    int id1 = tet_vert_ids[4 * n + 1];
    int id2 = tet_vert_ids[4 * n + 2];
    int id3 = tet_vert_ids[4 * n + 3];

    float vec_1[3] = {0.0, 0.0, 0.0};
    float vec_2[3] = {0.0, 0.0, 0.0};
    float vec_3[3] = {0.0, 0.0, 0.0};

    sub_vectors(&v_positions[id1 * 3], &v_positions[id0 * 3], vec_1);
    sub_vectors(&v_positions[id2 * 3], &v_positions[id0 * 3], vec_2);
    sub_vectors(&v_positions[id3 * 3], &v_positions[id0 * 3], vec_3);

    return tet_volume(vec_1, vec_2, vec_3);
}

__device__ int is_point_in_tri(float* p0, float* p1, float* p2, float* t)
{
    float vec01[3] = {0.0, 0.0, 0.0};
    float vec02[3] = {0.0, 0.0, 0.0};
    float vec0t[3] = {0.0, 0.0, 0.0};

    sub_vectors(p1, p0, vec01);
    sub_vectors(p2, p0, vec02);
    sub_vectors(t, p0, vec0t);

    float cross_total[3] = {0.0, 0.0, 0.0};
    cross_product_of(vec01, vec02, cross_total);
    float area_total = sqrtf(sqrd_length_of(cross_total)) / 2.0;

    float cross_1[3] = {0.0, 0.0, 0.0};
    cross_product_of(vec01, vec0t, cross_1);
    float area_1 = sqrtf(sqrd_length_of(cross_1)) / 2.0;

    float cross_2[3] = {0.0, 0.0, 0.0};
    cross_product_of(vec0t, vec02, cross_2);
    float area_2 = sqrtf(sqrd_length_of(cross_2)) / 2.0;

    if(dot_product_of(cross_1, cross_total) < 0) area_1 *= -1.0;
    if(dot_product_of(cross_2, cross_total) < 0) area_2 *= -1.0;

    if(area_1 >= 0 && area_2 >= 0 && area_1 + area_2 <= area_total)
    {
        return 1;
    }
    return 0;
}

__device__ int is_point_in_tet(float* p0, float* p1, float* p2, float* p3, float* t)
{
    float vec01[3] = {0.0, 0.0, 0.0};
    float vec02[3] = {0.0, 0.0, 0.0};
    float vec03[3] = {0.0, 0.0, 0.0};
    float vec0t[3] = {0.0, 0.0, 0.0};

    sub_vectors(p1, p0, vec01);
    sub_vectors(p2, p0, vec02);
    sub_vectors(p3, p0, vec03);
    sub_vectors(t, p0, vec0t);

    float vol = tet_volume(vec01, vec02, vec03);
    float vol1 = tet_volume(vec01, vec02, vec0t);
    float vol2 = tet_volume(vec02, vec03, vec0t);
    float vol3 = tet_volume(vec03, vec01, vec0t);

    if(vol1 >= 0 && vol2 >= 0 && vol3 >= 0 && vol1 + vol2 + vol3 <= vol)
    {
        return 1;
    }
    return 0;
}

__device__ int ray_plane_tracer(
    float* plane_point, 
    float* plane_normal, 
    float* ray_origin, 
    float* ray_dir, 
    float* intersection_out
)
{
    float plane_normal_normalized[3] = {0.0, 0.0, 0.0};
    float ray_dir_normalized[3] = {0.0, 0.0, 0.0};
    normalize_vector_to(plane_normal, plane_normal_normalized);
    normalize_vector_to(ray_dir, ray_dir_normalized);

    float ray_to_plane[3] = {0.0, 0.0, 0.0};
    sub_vectors(plane_point, ray_origin, ray_to_plane);

    float a = dot_product_of(ray_to_plane, plane_normal_normalized);
    float b = dot_product_of(ray_dir_normalized, plane_normal_normalized);
    if(b == 0) return 0;
    float t = a / b;
    if(t < 0) return 0;

    float intersection_point[3] = {0.0, 0.0, 0.0};
    scale_vector(ray_dir_normalized, intersection_point, t);
    add_vectors(intersection_point, ray_origin, intersection_out);
    return 1;
}

__device__ int triangle_ray_tracer(
    float* p0, 
    float* p1, 
    float* p2, 
    float* ray_origin, 
    float* ray_dir, 
    float* intersection_out
)
{
    float vec1[3] = {0.0, 0.0, 0.0};
    float vec2[3] = {0.0, 0.0, 0.0};
    sub_vectors(p1, p0, vec1);
    sub_vectors(p2, p0, vec2);

    float cross[3] = {0.0, 0.0, 0.0};
    cross_product_of(vec1, vec2, cross);

    float intersection[3] = {0.0, 0.0, 0.0};
    ray_plane_tracer(p0, cross, ray_origin, ray_dir, intersection);

    if(is_point_in_tri(p0, p1, p2, intersection))
    {
        copy_vector_to(intersection, intersection_out);
        return 1;
    } 
    return 0;
}

__device__ int tet_ray_tracer(
    float* p0, 
    float* p1, 
    float* p2, 
    float* ray_origin, 
    float* ray_dir, 
    float* intersection_out
)
{
    int tet_triangle_v_order[4][3] = {{1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}};
    for(int i = 0; i < 4; i++)
    {
    }
}

__device__ int get_mesh_offset(int* counts, int mesh_ID)
{
    int offset = 0;
    if(mesh_ID <= 0) return 0;
    for(int i = 0; i < mesh_ID; i++)
    {
        offset += counts[i];
    }
    return offset;
}
#endif /* VECMATH_CUH_ */