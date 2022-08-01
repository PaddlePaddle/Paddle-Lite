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

struct SoftmaxParam {
    int N;
    int K;
};

struct SoftmaxParam2 {
    int N;
    int C;
    int H;
    int W;
};

kernel void softmax(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant SoftmaxParam2& sp[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    int group = sp.N * sp.C / 4;
    int remain = sp.N * sp.C % 4;
#if LITE_WITH_METAL_FULL
    ftype max_value = FLT_MIN;
#else
    ftype max_value = HALF_MIN;
#endif

    // find max value
    for (int z = 0; z < group; z++) {
        ftype4 v = inTexture.read(uint2(gid.x, gid.y), z);
        max_value = max(v[0], max_value);
        max_value = max(v[1], max_value);
        max_value = max(v[2], max_value);
        max_value = max(v[3], max_value);
    }
    if (remain > 0) {
        ftype4 r = inTexture.read(uint2(gid.x, gid.y), group);
        for (int i = 0; i < remain; i++) {
            max_value = max(r[i], max_value);
        }
    }

    // need to consider the numerical error of overﬂow and underﬂow
    // doc: https://www.deeplearningbook.org/contents/numerical.html
    ftype4 sumv = {0, 0, 0, 0};
    for (int z = 0; z < group; z++) {
        ftype4 v = inTexture.read(uint2(gid.x, gid.y), z);
        sumv += exp(v - max_value);
    }
    ftype sum = sumv.x + sumv.y + sumv.z + sumv.w;
    if (remain > 0) {
        ftype4 r = inTexture.read(uint2(gid.x, gid.y), group);
        for (int i = 0; i < remain; i++) {
            sum += exp(r[i] - max_value);
        }
    }
    for (int z = 0; z <= group; z++) {
        ftype4 v = inTexture.read(uint2(gid.x, gid.y), z);
        v = exp(v - max_value);
        outTexture.write(v / sum, gid.xy, z);
    }
}

kernel void softmax_c_d3_common(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant SoftmaxParam2& sp[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    int out_texture_array_size = outTexture.get_array_size();
    int left = out_texture_array_size * 4 - sp.N * sp.C;

    ftype4 max_vector = inTexture.read(uint2(gid.x, gid.y), 0);

    // caculate max
    int array_size = inTexture.get_array_size();
    for (int z = 0; z < (array_size - 1); z++) {
        ftype4 temp_value_vector = inTexture.read(uint2(gid.x, gid.y), z);
        max_vector = max(temp_value_vector, max_vector);
    }

    ftype max_value = max_vector[0];
    if (array_size > 1) {
        for (int c = 0; c < 4; c++) {
            max_value = max(max_vector[c], max_value);
        }
    }

    ftype4 temp_value_vector = inTexture.read(uint2(gid.x, gid.y), array_size - 1);
    ftype max_value_left = temp_value_vector[0];
    for (int c = 0; c < left; c++) {
        max_value_left = max(temp_value_vector[c], max_value_left);
    }
    max_value = max(max_value, max_value_left);

    // caculate sum
    ftype4 sum_vector = 0.0;
    for (int z = 0; z < array_size - 1; z++) {
        ftype4 temp_value_vector = inTexture.read(uint2(gid.x, gid.y), z);
        sum_vector += exp(temp_value_vector - max_value);
    }
    ftype sum_value = 0.0;
    if (array_size > 1) {
        sum_value = sum_vector[0] + sum_vector[1] + sum_vector[2] + sum_vector[3];
    }
    ftype4 sum_vector_left = 0.0;
    ftype4 temp_value_vector_left = inTexture.read(uint2(gid.x, gid.y), array_size - 1);
    sum_vector_left += exp(temp_value_vector_left - max_value);

    ftype sum_value_left = 0.0;
    for (int i = 0; i < left; i++) {
        sum_value_left += sum_vector_left[i];
    }
    sum_value += sum_value_left;

    // calculate output
    ftype4 result_vector = inTexture.read(gid.xy, gid.z);
    result_vector = exp(result_vector - max_value) / sum_value;
    outTexture.write(result_vector, gid.xy, gid.z);
}

kernel void softmax_w_d3_common(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant SoftmaxParam2& sp[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    // caculate sum
    ftype4 max_vector = inTexture.read(uint2(0, gid.y), gid.z);
    int w = sp.W;
    for (int x = 1; x < w; x++) {
        ftype4 temp_value_vector = inTexture.read(uint2(x, gid.y), gid.z);
        max_vector = max(temp_value_vector, max_vector);
    }

    // caculate sum
    ftype4 sum_vector = 0.0;
    for (int x = 0; x < w; x++) {
        ftype4 temp_value_vector = inTexture.read(uint2(x, gid.y), gid.z);
        sum_vector += exp(temp_value_vector - max_vector);
    }

    // calculate output
    ftype4 result_vector = inTexture.read(gid.xy, gid.z);
    result_vector = exp(result_vector - max_vector) / sum_vector;
    outTexture.write(result_vector, gid.xy, gid.z);
}

kernel void softmax_h_d3_common(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant SoftmaxParam2& sp[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    ftype4 max_vector = inTexture.read(uint2(gid.x, 0), gid.z);

    // caculate max
    int h = sp.H;
    for (int y = 1; y < h; y++) {
        ftype4 temp_value_vector = inTexture.read(uint2(gid.x, y), gid.z);
        max_vector = max(temp_value_vector, max_vector);
    }

    // caculate sum
    ftype4 sum_vector = 0.0;
    for (int y = 0; y < h; y++) {
        ftype4 temp_value_vector = inTexture.read(uint2(gid.x, y), gid.z);
        sum_vector += exp(temp_value_vector - max_vector);
    }

    // calculate output
    ftype4 result_vector = inTexture.read(gid.xy, gid.z);
    result_vector = exp(result_vector - max_vector) / sum_vector;
    outTexture.write(result_vector, gid.xy, gid.z);
}

kernel void softmax_dim2_common(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant SoftmaxParam2& sp[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    // caculate sum
    ftype4 max_vector = 0;
    int w = sp.W % 4 == 0 ? sp.W / 4 : sp.W / 4 + 1;
    int res = sp.W % 4 == 0 ? 4 : sp.W % 4;
    ftype max_num = 0;
    ftype sum_num = 0;
    for (int z = 0; z < w; z++) {
        ftype4 temp_value_vector = inTexture.read(uint2(gid.x, gid.y), z);
        max_vector = max(temp_value_vector, max_vector);
    }
    for (int z = 0; z < 4; z++) {
        max_num = max(max_vector[z], max_num);
    }

    // caculate sum
    ftype4 sum_vector = 0.0;
    for (int z = 0; z < w - 1; z++) {
        ftype4 tem_value_vector = inTexture.read(uint2(gid.x, gid.y), z);
        sum_vector += exp(tem_value_vector - max_num);
    }
    for (int z = 0; z < 4; z++) {
        sum_num += sum_vector[z];
    }
    ftype4 tem_value_vector = inTexture.read(uint2(gid.x, gid.y), w - 1);
    for (int z = 0; z < res; z++) {
        sum_num += exp(tem_value_vector[z] - max_num);
    }

    // calculate output
    ftype4 result_vector = inTexture.read(uint2(gid.x, gid.y), gid.z);
    result_vector = exp(result_vector - max_num) / sum_num;
    outTexture.write(result_vector, uint2(gid.x, gid.y), gid.z);
}
