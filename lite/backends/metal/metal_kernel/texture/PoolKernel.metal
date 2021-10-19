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

#include "Common.metal"

using namespace metal;

struct PoolParam {
    int ksizeX;
    int ksizeY;
    int strideX;
    int strideY;
    int paddingX;
    int paddingY;
    int poolType;
    int exclusive;
};

kernel void pool(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant PoolParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    int xmin = gid.x * pm.strideX - pm.paddingX;
    int xmax = min(xmin + pm.ksizeX, int(inTexture.get_width()));
    xmin = max(xmin, 0);
    int ymin = gid.y * pm.strideY - pm.paddingY;
    int ymax = min(ymin + pm.ksizeY, int(inTexture.get_height()));
    ymin = max(ymin, 0);

    ftype4 r = 0;
    if (pm.poolType == 0) {
        r = inTexture.read(uint2(xmin, ymin), gid.z);
        for (int x = xmin; x < xmax; x++) {
            for (int y = ymin; y < ymax; y++) {
                r = fmax(r, inTexture.read(uint2(x, y), gid.z));
            }
        }
    } else if (pm.poolType == 1) {
        for (int x = xmin; x < xmax; x++) {
            for (int y = ymin; y < ymax; y++) {
                r += inTexture.read(uint2(x, y), gid.z);
            }
        }
        int count = pm.exclusive ? (xmax - xmin) * (ymax - ymin) : (pm.ksizeY * pm.ksizeX);
        ftype4 div = count > 0 ? 1.f / count : 0.0;
        r *= div;
    }
    outTexture.write(r, gid.xy, gid.z);
}

kernel void global_pool(texture2d_array<ftype, access::read> in[[texture(0)]],
    texture2d_array<ftype, access::write> out[[texture(1)]],
    ushort3 gid[[thread_position_in_grid]],
    ushort tid[[thread_index_in_threadgroup]],
    ushort3 tg_size[[threads_per_threadgroup]]) {
    ushort width = in.get_width();
    ushort height = in.get_height();
    const ushort thread_count = tg_size.x * tg_size.y;

    threadgroup ftype4 shared_mem[256];

    ftype4 sum = 0;
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

    const ftype4 mean = shared_mem[0];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    out.write(mean, uint2(0, 0), gid.z);
}
