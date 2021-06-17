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

#include <metal_stdlib>
#include "Common.metal"
using namespace metal;

struct bilinear_interp_param {
  float ratio_h;
  float ratio_w;
  float align_delta;
};

kernel void bilinear_interp(texture2d_array<ftype, access::read> input [[texture(0)]],
                    texture2d_array<ftype, access::write> output [[texture(1)]],
                    constant bilinear_interp_param& pm [[buffer(0)]],
                    uint3 gid [[thread_position_in_grid]]) {
  ftype4 r;
  if ((input.get_width() == output.get_width()) &&
      (input.get_height() == output.get_height())) {
    r = input.read(gid.xy, gid.z);
  } else {
    ftype w = (gid.x + pm.align_delta) * pm.ratio_w - pm.align_delta;
    ftype h = (gid.y + pm.align_delta) * pm.ratio_h - pm.align_delta;
    uint w0 = w, h0 = h;
    uint w1 = w0 + 1, h1 = h0 + 1;

    ftype w1lambda = w - w0, h1lambda = h - h0;
    ftype w2lambda = 1.0 - w1lambda, h2lambda = 1.0 - h1lambda;
    if (w1 >= input.get_width()) {
      w1 = w0;
    }
    if (h1 >= input.get_height()) {
      h1 = h0;
    }
    ftype4 r0 = input.read(uint2(w0, h0), gid.z);  // bottom left
    ftype4 r1 = input.read(uint2(w1, h0), gid.z);  // bottom right
    ftype4 r2 = input.read(uint2(w0, h1), gid.z);  // top left
    ftype4 r3 = input.read(uint2(w1, h1), gid.z);  // top right

    r = h2lambda * (w2lambda * r0 + w1lambda * r1) +
        h1lambda * (w2lambda * r2 + w1lambda * r3);
  }
  output.write(r, gid.xy, gid.z);
}
