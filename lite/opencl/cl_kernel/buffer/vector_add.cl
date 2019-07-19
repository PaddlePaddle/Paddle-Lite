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

__kernel void vector_add (__global const float* src_a,
                     __global const float* src_b,
                     __global float* res,
                     const int num)
{
   const int idx = get_global_id(0);
   if (idx < num)
      res[idx] = src_a[idx] + src_b[idx];
}