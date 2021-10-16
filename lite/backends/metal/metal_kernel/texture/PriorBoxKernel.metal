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

struct PriorBoxMetalParam {
    float offset;
    float stepWidth;
    float stepHeight;
    float minSize;
    float maxSize;
    float imageWidth;
    float imageHeight;

    bool clip;

    uint numPriors;
    uint aspecRatiosSize;
    uint minSizeSize;
    uint maxSizeSize;
};

kernel void prior_box(texture2d_array<float, access::read> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outBoxTexture[[texture(1)]],
    texture2d_array<float, access::write> varianceTexture[[texture(2)]],
    const device float* aspect_ratios[[buffer(0)]],
    constant PriorBoxMetalParam& param[[buffer(1)]],
    const device float4* variances[[buffer(2)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outBoxTexture.get_width() || gid.y >= outBoxTexture.get_height() ||
        gid.z >= outBoxTexture.get_array_size())
        return;

    float center_x = (gid.x + param.offset) * param.stepWidth;
    float center_y = (gid.y + param.offset) * param.stepHeight;

    float box_width, box_height;

    if (gid.z < param.aspecRatiosSize) {
        float ar = aspect_ratios[gid.z];
        box_width = param.minSize * sqrt(ar) / 2;
        box_height = param.minSize / sqrt(ar) / 2;
        float4 box;
        box.x = (center_x - box_width) / param.imageWidth;
        box.y = (center_y - box_height) / param.imageHeight;
        box.z = (center_x + box_width) / param.imageWidth;
        box.w = (center_y + box_height) / param.imageHeight;

        float4 res;
        if (param.clip) {
            res = fmin(fmax(box, 0.0), 1.0);
        } else {
            res = box;
        }

        outBoxTexture.write(res, gid.xy, gid.z);
    } else if (gid.z >= param.aspecRatiosSize) {
        if (param.maxSizeSize > 0) {
            box_width = box_height = sqrt(param.minSize * param.maxSize) / 2;
            float4 max_box;
            max_box.x = (center_x - box_width) / param.imageWidth;
            max_box.y = (center_y - box_height) / param.imageHeight;
            max_box.z = (center_x + box_width) / param.imageWidth;
            max_box.w = (center_y + box_height) / param.imageHeight;

            float4 res;
            if (param.clip) {
                res = min(max(max_box, 0.0), 1.0);
            } else {
                res = max_box;
            }
            outBoxTexture.write(max_box, gid.xy, gid.z);
        }
    }

    float4 variance = variances[0];
    if (gid.z < param.numPriors) {
        float4 variances_output;
        variances_output.x = variance.x;
        variances_output.y = variance.y;
        variances_output.z = variance.z;
        variances_output.w = variance.w;
        varianceTexture.write(variances_output, gid.xy, gid.z);
    }
}

kernel void prior_box_half(texture2d_array<half, access::read> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outBoxTexture[[texture(1)]],
    texture2d_array<half, access::write> varianceTexture[[texture(2)]],
    const device half* aspect_ratios[[buffer(0)]],
    constant PriorBoxMetalParam& param[[buffer(1)]],
    const device float4* variances[[buffer(2)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outBoxTexture.get_width() || gid.y >= outBoxTexture.get_height() ||
        gid.z >= outBoxTexture.get_array_size())
        return;

    float center_x = (gid.x + param.offset) * param.stepWidth;
    float center_y = (gid.y + param.offset) * param.stepHeight;

    float box_width, box_height;

    if (gid.z < param.aspecRatiosSize) {
        half ar = aspect_ratios[gid.z];
        box_width = param.minSize * sqrt(ar) / 2;
        box_height = param.minSize / sqrt(ar) / 2;
        float4 box;
        box.x = (center_x - box_width) / param.imageWidth;
        box.y = (center_y - box_height) / param.imageHeight;
        box.z = (center_x + box_width) / param.imageWidth;
        box.w = (center_y + box_height) / param.imageHeight;

        float4 res;
        if (param.clip) {
            res = fmin(fmax(box, 0.0), 1.0);
        } else {
            res = box;
        }

        outBoxTexture.write(half4(res), gid.xy, gid.z);
    } else if (gid.z >= param.aspecRatiosSize) {
        if (param.maxSizeSize > 0) {
            box_width = box_height = sqrt(param.minSize * param.maxSize) / 2;
            float4 max_box;
            max_box.x = (center_x - box_width) / param.imageWidth;
            max_box.y = (center_y - box_height) / param.imageHeight;
            max_box.z = (center_x + box_width) / param.imageWidth;
            max_box.w = (center_y + box_height) / param.imageHeight;

            float4 res;
            if (param.clip) {
                res = min(max(max_box, 0.0), 1.0);
            } else {
                res = max_box;
            }
            outBoxTexture.write(half4(max_box), gid.xy, gid.z);
        }
    }

    float4 variance = variances[0];
    if (gid.z < param.numPriors) {
        float4 variances_output;
        variances_output.x = variance.x;
        variances_output.y = variance.y;
        variances_output.z = variance.z;
        variances_output.w = variance.w;
        varianceTexture.write(half4(variances_output), gid.xy, gid.z);
    }
}

kernel void prior_box_MinMaxAspectRatiosOrder(
    texture2d_array<float, access::read> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outBoxTexture[[texture(1)]],
    texture2d_array<float, access::write> varianceTexture[[texture(2)]],
    const device float* aspect_ratios[[buffer(0)]],
    constant PriorBoxMetalParam& param[[buffer(1)]],
    const device float4* variances[[buffer(2)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outBoxTexture.get_width() || gid.y >= outBoxTexture.get_height() ||
        gid.z >= outBoxTexture.get_array_size())
        return;

    float center_x = (gid.x + param.offset) * param.stepWidth;
    float center_y = (gid.y + param.offset) * param.stepHeight;

    float box_width, box_height;

    if (gid.z == 0) {
        box_width = box_height = param.minSize / 2;

        float4 box;
        box.x = (center_x - box_width) / param.imageWidth;
        box.y = (center_y - box_height) / param.imageHeight;
        box.z = (center_x + box_width) / param.imageWidth;
        box.w = (center_y + box_height) / param.imageHeight;

        float4 res;
        if (param.clip) {
            res = fmin(fmax(box, 0.0), 1.0);
        } else {
            res = box;
        }

        outBoxTexture.write(res, gid.xy, gid.z);
    }

    if (gid.z == 1 && param.maxSizeSize > 0) {
        box_width = box_height = sqrt(param.minSize * param.maxSize) / 2;
        float4 max_box;
        max_box.x = (center_x - box_width) / param.imageWidth;
        max_box.y = (center_y - box_height) / param.imageHeight;
        max_box.z = (center_x + box_width) / param.imageWidth;
        max_box.w = (center_y + box_height) / param.imageHeight;

        float4 res;
        if (param.clip) {
            res = min(max(max_box, 0.0), 1.0);
        } else {
            res = max_box;
        }
        outBoxTexture.write(res, gid.xy, gid.z);
    }

    int aspect_to = 0;
    if (param.maxSizeSize > 0) {
        aspect_to = gid.z - 2;
    } else {
        aspect_to = gid.z - 1;
    }

    if (aspect_to >= 0 && aspect_to < int(param.aspecRatiosSize)) {
        int skip = 0;
        for (int i = 0; i < aspect_to + 1; ++i) {
            if (fabs(aspect_ratios[i] - 1.) < 1e-6) {
                skip += 1;
            }
        }
        aspect_to += skip;

        float ar = aspect_ratios[aspect_to];

        box_width = param.minSize * sqrt(ar) / 2;
        box_height = param.minSize / sqrt(ar) / 2;
        float4 box;
        box.x = (center_x - box_width) / param.imageWidth;
        box.y = (center_y - box_height) / param.imageHeight;
        box.z = (center_x + box_width) / param.imageWidth;
        box.w = (center_y + box_height) / param.imageHeight;

        float4 res;
        if (param.clip) {
            res = fmin(fmax(box, 0.0), 1.0);
        } else {
            res = box;
        }

        outBoxTexture.write(res, gid.xy, gid.z);
    }

    float4 variance = variances[0];
    if (gid.z < param.numPriors) {
        float4 variances_output;
        variances_output.x = variance.x;
        variances_output.y = variance.y;
        variances_output.z = variance.z;
        variances_output.w = variance.w;
        varianceTexture.write(variances_output, gid.xy, gid.z);
    }
}

kernel void prior_box_MinMaxAspectRatiosOrder_half(
    texture2d_array<half, access::read> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outBoxTexture[[texture(1)]],
    texture2d_array<half, access::write> varianceTexture[[texture(2)]],
    const device half* aspect_ratios[[buffer(0)]],
    constant PriorBoxMetalParam& param[[buffer(1)]],
    const device float4* variances[[buffer(2)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outBoxTexture.get_width() || gid.y >= outBoxTexture.get_height() ||
        gid.z >= outBoxTexture.get_array_size())
        return;

    float center_x = (gid.x + param.offset) * param.stepWidth;
    float center_y = (gid.y + param.offset) * param.stepHeight;

    float box_width, box_height;

    if (gid.z == 0) {
        box_width = box_height = param.minSize / 2;

        float4 box;
        box.x = (center_x - box_width) / param.imageWidth;
        box.y = (center_y - box_height) / param.imageHeight;
        box.z = (center_x + box_width) / param.imageWidth;
        box.w = (center_y + box_height) / param.imageHeight;

        float4 res;
        if (param.clip) {
            res = fmin(fmax(box, 0.0), 1.0);
        } else {
            res = box;
        }

        outBoxTexture.write(half4(res), gid.xy, gid.z);
    }

    if (gid.z == 1 && param.maxSizeSize > 0) {
        box_width = box_height = sqrt(param.minSize * param.maxSize) / 2;
        float4 max_box;
        max_box.x = (center_x - box_width) / param.imageWidth;
        max_box.y = (center_y - box_height) / param.imageHeight;
        max_box.z = (center_x + box_width) / param.imageWidth;
        max_box.w = (center_y + box_height) / param.imageHeight;

        float4 res;
        if (param.clip) {
            res = min(max(max_box, 0.0), 1.0);
        } else {
            res = max_box;
        }
        outBoxTexture.write(half4(res), gid.xy, gid.z);
    }

    int aspect_to = 0;
    if (param.maxSizeSize > 0) {
        aspect_to = gid.z - 2;
    } else {
        aspect_to = gid.z - 1;
    }

    if (aspect_to > 0 && aspect_to < int(param.aspecRatiosSize) &&
        fabs(aspect_ratios[aspect_to] - 1.) > 1e-6) {
        float ar = aspect_ratios[aspect_to];

        box_width = param.minSize * sqrt(ar) / 2;
        box_height = param.minSize / sqrt(ar) / 2;
        float4 box;
        box.x = (center_x - box_width) / param.imageWidth;
        box.y = (center_y - box_height) / param.imageHeight;
        box.z = (center_x + box_width) / param.imageWidth;
        box.w = (center_y + box_height) / param.imageHeight;

        float4 res;
        if (param.clip) {
            res = fmin(fmax(box, 0.0), 1.0);
        } else {
            res = box;
        }

        outBoxTexture.write(half4(res), gid.xy, gid.z);
    }

    float4 variance = variances[0];
    if (gid.z < param.numPriors) {
        float4 variances_output;
        variances_output.x = variance.x;
        variances_output.y = variance.y;
        variances_output.z = variance.z;
        variances_output.w = variance.w;
        varianceTexture.write(half4(variances_output), gid.xy, gid.z);
    }
}
