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

#pragma once

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Note: macro name replacement need twice parser
#define GET_VEC_TYPE(type__, size__) type__##size__
#define VECTORIZED_TYPE(type__, size__) GET_VEC_TYPE(type__, size__)
#define CL_DTYPE4 VECTORIZED_TYPE(CL_DTYPE, 4)

#define _CONVERT_TYPE_TO(value, type) convert_##type(value)
#define CONVERT_TYPE_TO(value, type) _CONVERT_TYPE_TO(value, type)

#define _WRITE_IMG_TYPE(type_char, img, pos, value) \
  write_image##type_char(img, pos, value)
#define WRITE_IMG_TYPE(type_char, img, pos, value) \
  _WRITE_IMG_TYPE(type_char, img, pos, value)

#define _READ_IMG_TYPE(type_char, img, pos, sampler) \
  read_image##type_char(img, sampler, pos)
#define READ_IMG_TYPE(type_char, img, pos, sampler) \
  _READ_IMG_TYPE(type_char, img, pos, sampler)

inline CL_DTYPE activation(CL_DTYPE in
#ifdef PRELU
                           ,
                           CL_DTYPE prelu_alpha
#endif
                           ) {
  CL_DTYPE output;
#ifdef PRELU
  output = select(prelu_alpha * in, in, in >= (CL_DTYPE)0);
#endif

#ifdef RELU
  output = fmax(in, (CL_DTYPE)0);
#endif
  return output;
}
