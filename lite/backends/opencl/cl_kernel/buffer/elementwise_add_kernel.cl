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

__kernel void elementwise_add(__global const CL_DTYPE* x_data,
                              __global const CL_DTYPE* y_data,
                              __global CL_DTYPE* out_data,
                              const int batch,
                              const int channels,
                              const int num) {
  const int c = get_global_id(0);  // c: [0, channels)
  const int b = get_global_id(1);  // b: [0, batch)

  if ((c >= channels) || (b >= batch)) {
    return;
  }

  const int offset = (b * channels + c) * num;

  __global const CL_DTYPE* din_ptr = x_data + offset;
  const CL_DTYPE diny_data = y_data[c];
  __global CL_DTYPE* dout_ptr = out_data + offset;

  for (int n = 0; n < num; ++n) {  // n: [0, h*w)
    *dout_ptr = *din_ptr + diny_data;
#ifdef RELU
    CL_DTYPE alpha;
    *dout_ptr = activation(*dout_ptr, alpha);
#endif
    ++dout_ptr;
    ++din_ptr;
  }
}
