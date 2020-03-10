/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cl_common.h>


__kernel void instance_norm(__read_only image2d_t input,
                            __write_only image2d_t output,
                            __read_only image2d_t scale,
                            __read_only image2d_t bias,
                            const float epsilon,
                            const int in_h,
                            const int in_w){
    __local CL_DTYPE4 saved_mean[1024];
    __local CL_DTYPE4 saved_variance[1024];
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int gidx = get_group_id(0);
    const int gidy = get_group_id(1);
    const int spatial_size = in_h * in_w;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
    CL_DTYPE4 mean = (CL_DTYPE4)(0.f, 0.f, 0.f, 0.f);
    CL_DTYPE4 variance = (CL_DTYPE4)(0.f, 0.f, 0.f, 0.f);
    CL_DTYPE4 vepsilon = (CL_DTYPE4)(epsilon, epsilon, epsilon, epsilon);
    const int x_offset = gidx * in_w;
    const int y_offset = gidy * in_h;
    int2 coor;
    for (int i = lid; i < spatial_size; i += lsize) {
        coor.x = i % in_w + x_offset;
        coor.y = i / in_w + y_offset;
        CL_DTYPE4 pixel = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coor);
        mean += pixel;
        variance += pixel * pixel;
    }
    saved_mean[lid] = mean;
    saved_variance[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    //! do reduction
    int dynamic_size = lsize >> 1;
    for (; dynamic_size > 0; dynamic_size >>= 1){
        if (lid < dynamic_size) {
          saved_mean[lid] += saved_mean[lid + dynamic_size];
          saved_variance[lid] += saved_variance[lid + dynamic_size];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    mean = saved_mean[0] / spatial_size;
    variance = saved_variance[0] / spatial_size - mean * mean;
    variance = rsqrt(variance + vepsilon);
    
    //! do instance norm
    coor.x = gidx;
    coor.y = gidy;
    CL_DTYPE4 vscale = READ_IMG_TYPE(CL_DTYPE_CHAR, scale, sampler, coor);
    vscale *= variance;
    CL_DTYPE4 vbias = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coor);
    for (int i = lid; i < spatial_size; i += lsize) {
        coor.x = i % in_w + x_offset;
        coor.y = i / in_w + y_offset;
        CL_DTYPE4 pixel = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coor);
        pixel = (pixel - mean) * vscale + vbias;
        WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, coor, pixel);
    }
}
