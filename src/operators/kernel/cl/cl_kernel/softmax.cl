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

__kernel void softmax(__read_only image2d_t input,
                      __write_only image2d_t output,
                      __private const int d0,
                      __private const int d1,
                      __private const int d2,
                      __private const int d3) {
  const int z = get_global_id(0);
  const int x = get_global_id(1);
  const int y = get_global_id(2);
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
  half4 maxv = read_imageh(input, sampler, int2(z * d3, y));
  half4 buf[d3] = {piece};
  for (int i = 1; i < d3; i++) {
    buf[i] = read_imageh(input, sampler, int2(z * d3 + i, y));
    maxv = max(maxv, buf[i]);
  }
  float4 sum = 0;
  for (int i = 0; i < d3; i++) {
    buf[i] = exp(buf[i] - maxv);
    sum += buf[i];
  }
  half4 r = buf[x] / sum;

  write_imageh(output, int2(z * d3 + x, y), r);
}
