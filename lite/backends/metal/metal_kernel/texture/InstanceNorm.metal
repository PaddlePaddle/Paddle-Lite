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

#include <metal_stdlib>
using namespace metal;

kernel void instance_norm(texture2d_array<float, access::read> in[[texture(0)]],
    texture2d_array<float, access::write> out[[texture(1)]],

    ushort3 gid[[thread_position_in_grid]],
    ushort tid[[thread_index_in_threadgroup]],
    ushort3 tg_size[[threads_per_threadgroup]]) {
    ushort width = in.get_width();
    ushort height = in.get_height();
    const ushort thread_count = tg_size.x * tg_size.y;

    threadgroup float4 shared_mem[256];

    float4 sum = 0;
    for (ushort xIndex = gid.x; xIndex < width; xIndex += tg_size.x) {
        for (ushort yIndex = gid.y; yIndex < height; yIndex += tg_size.y) {
            sum += in.read(uint2(xIndex, yIndex), gid.z);
        }
    }
    shared_mem[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = 0;
    if (tid < 32) {
        for (ushort i = tid + 32; i < thread_count; i += 32) {
            sum += shared_mem[i];
        }
    }
    shared_mem[tid] += sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = 0;
    if (tid == 0) {
        ushort top = min(ushort(32), thread_count);
        for (ushort i = 0; i < top; i += 1) {
            sum += shared_mem[i];
        }
        shared_mem[0] = sum / (width * height);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float4 mean = shared_mem[0];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = 0;
    for (ushort xIndex = gid.x; xIndex < width; xIndex += tg_size.x) {
        for (ushort yIndex = gid.y; yIndex < height; yIndex += tg_size.y) {
            sum += pow(in.read(uint2(xIndex, yIndex), gid.z) - mean, 2);
        }
    }

    shared_mem[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = 0;
    if (tid < 32) {
        for (ushort i = tid + 32; i < thread_count; i += 32) {
            sum += shared_mem[i];
        }
    }
    shared_mem[tid] += sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = 0;
    if (tid == 0) {
        ushort top = min(ushort(32), thread_count);
        for (ushort i = 0; i < top; i += 1) {
            sum += shared_mem[i];
        }
        shared_mem[0] = sum / (width * height);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float4 sigma = sqrt(shared_mem[0] + float4(1e-5));

    float4 multiplier = 1 / sigma;
    for (ushort xIndex = gid.x; xIndex < width; xIndex += tg_size.x) {
        for (ushort yIndex = gid.y; yIndex < height; yIndex += tg_size.y) {
            float4 val = in.read(uint2(xIndex, yIndex), gid.z);
            out.write((val - mean) * multiplier + 0, uint2(xIndex, yIndex), gid.z);
        }
    }
}

kernel void instance_norm_half(texture2d_array<half, access::read> in[[texture(0)]],
    texture2d_array<half, access::write> out[[texture(1)]],

    ushort3 gid[[thread_position_in_grid]],
    ushort tid[[thread_index_in_threadgroup]],
    ushort3 tg_size[[threads_per_threadgroup]]) {
    ushort width = in.get_width();
    ushort height = in.get_height();
    const ushort thread_count = tg_size.x * tg_size.y;

    threadgroup float4 shared_mem[256];

    float4 sum = 0;
    for (ushort xIndex = gid.x; xIndex < width; xIndex += tg_size.x) {
        for (ushort yIndex = gid.y; yIndex < height; yIndex += tg_size.y) {
            sum += float4(in.read(uint2(xIndex, yIndex), gid.z));
        }
    }
    shared_mem[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = 0;
    if (tid < 32) {
        for (ushort i = tid + 32; i < thread_count; i += 32) {
            sum += shared_mem[i];
        }
    }
    shared_mem[tid] += sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = 0;
    if (tid == 0) {
        ushort top = min(ushort(32), thread_count);
        for (ushort i = 0; i < top; i += 1) {
            sum += shared_mem[i];
        }
        shared_mem[0] = sum / (width * height);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float4 mean = shared_mem[0];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = 0;
    for (ushort xIndex = gid.x; xIndex < width; xIndex += tg_size.x) {
        for (ushort yIndex = gid.y; yIndex < height; yIndex += tg_size.y) {
            sum += pow(float4(in.read(uint2(xIndex, yIndex), gid.z)) - mean, 2);
        }
    }

    shared_mem[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = 0;
    if (tid < 32) {
        for (ushort i = tid + 32; i < thread_count; i += 32) {
            sum += shared_mem[i];
        }
    }
    shared_mem[tid] += sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = 0;
    if (tid == 0) {
        ushort top = min(ushort(32), thread_count);
        for (ushort i = 0; i < top; i += 1) {
            sum += shared_mem[i];
        }
        shared_mem[0] = sum / (width * height);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float4 sigma = sqrt(shared_mem[0] + float4(1e-5));

    float4 multiplier = 1 / sigma;
    for (ushort xIndex = gid.x; xIndex < width; xIndex += tg_size.x) {
        for (ushort yIndex = gid.y; yIndex < height; yIndex += tg_size.y) {
            float4 val = float4(in.read(uint2(xIndex, yIndex), gid.z));
            out.write(half4((val - mean) * multiplier + 0), uint2(xIndex, yIndex), gid.z);
        }
    }
}
