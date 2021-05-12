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

#ifdef P

#include "Macro.metal"

// 激活函数hard swish
// output = input * (min(max(0, input + offset), threshold)) / scale
// 具体算法详见文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn/hard_swish_cn.html#hard-swish
kernel void FUNC2_(hard_swish,
                   P)(texture2d_array<P, access::read> input [[texture(0)]],
                      texture2d_array<P, access::write> output [[texture(1)]],
                      constant HardSwishParam &pm [[buffer(0)]],
                      uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= output.get_width() || gid.y >= output.get_height() ||
      gid.z >= output.get_array_size()) {
    return;
  }

  VECTOR(P, 4) input_value = input.read(gid.xy, gid.z);
  VECTOR(P, 4)
  output_value =
      input_value * (min(max(0.0, input_value + pm.offset), pm.threshold)) /
      pm.scale;
  output.write(output_value, gid.xy, gid.z);
}

#endif
