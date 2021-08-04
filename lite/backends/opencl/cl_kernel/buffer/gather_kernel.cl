/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void gather_without_axis_nchw( __global const CL_DTYPE* input,
                                        __global CL_DTYPE* output,
                                        __global const CL_DTYPE* index,
                                        int slice_size,
                                        int batch_size,
                                        int height,
                                        int width) {
    const int h = get_global_id(0); // height
    const int w = get_global_id(1); // width
    const int c = get_global_id(2); // channel

    for (int i = 0; i < batch_size; ++i) {
        int ind = (int)index[i]; // for dim n
        int in_off = ind * slice_size + c * height * width + h * width + w;
        int out_off = i * slice_size + c * height * width + h * width + w;
        output[out_off] = input[in_off];
    }
}

__kernel void gather_without_axis_chw( __global const CL_DTYPE* input,
                                        __global CL_DTYPE* output,
                                        __global const CL_DTYPE* index,
                                        int slice_size,
                                        int index_size,
                                        int height,
                                        int width) {
    const int h = get_global_id(0); // height
    const int w = get_global_id(1); // width
    const int c = get_global_id(2); // index_size

    int ind = (int)index[c]; // for dim c
    int in_off = ind * height * width + h * width + w;
    int out_off = c * height * width + h * width + w;
    output[out_off] = input[in_off];
}

__kernel void gather_without_axis_hw( __global const CL_DTYPE* input,
                                        __global CL_DTYPE* output,
                                        __global const CL_DTYPE* index,
                                        int index_size,
                                        int width) {
    const int h = get_global_id(0); // index_size
    const int w = get_global_id(1); // width

    int ind = (int)index[h];
    int in_off = ind * width + w;
    int out_off = h * width + w;
    output[out_off] = input[in_off];
}

__kernel void gather_with_axis( __global const CL_DTYPE* input,
                                __global CL_DTYPE* output,
                                __global const CL_DTYPE* index,
                                int input_size,
                                int index_size,
                                int inner_dim_size,
                                int outer_dim_size ) {
    const int in = get_global_id(0); // inner_dim_size
    const int mid = get_global_id(1); // index_size
    const int out = get_global_id(2); // outer_dim_size

    int in_off = in * (input_size / inner_dim_size) + index[mid] * outer_dim_size + out;
    int out_off = in * index_size * outer_dim_size + mid * outer_dim_size + out;
    output[out_off] = input[in_off];
}