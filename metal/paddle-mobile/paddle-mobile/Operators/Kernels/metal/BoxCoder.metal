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
using namespace metal;

kernel void boxcoder(texture2d_array<float, access::read> priorBox [[texture(0)]],
                     texture2d_array<float, access::read> priorBoxVar [[texture(1)]],
                     texture2d_array<float, access::read> targetBox [[texture(2)]],
                     texture2d_array<float, access::write> output[[texture(3)]],
                     uint3 gid [[thread_position_in_grid]]) {
  float4 t = targetBox.read(gid.xy, gid.z);
  float4 p = priorBox.read(gid.xy, gid.z);
  float4 pv = priorBoxVar.read(gid.xy, gid.z);
  
  float px = (p.x + p.z) / 2;
  float py = (p.y + p.w) / 2;
  float pw = p.z - p.x;
  float ph = p.w - p.y;
  
  float tx = pv.x * t.x * pw + px;
  float ty = pv.y * t.y * ph + py;
  float tw = exp(pv.z * t.z) * pw;
  float th = exp(pv.w * t.w) * ph;
  
  
  float4 r;
  r.x = tx - tw / 2;
  r.y = ty - th / 2;
  r.z = tx + tw / 2;
  r.w = ty + th / 2;

  output.write(r, gid.xy, gid.z);
}
