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

/////////////////////////////////
// fp16 enabled, MAX_VALUE, MIN_VALUE
/////////////////////////////////
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define MAX_VALUE FLT_MAX
#define MIN_VALUE -FLT_MAX

/////////////////////////////////
// CL_DTYPE_float / CL_DTYPE_half
/////////////////////////////////
// Data type: pass one of macros on host: [CL_DTYPE_float, CL_DYPE_half]
#ifdef CL_DTYPE_float
#define CL_DTYPE float
#define CL_DTYPE_CHAR f
#define CL_COMPUTE_DTYPE half
#define CL_COMPUTE_DTYPE_CHAR h
#endif

#ifdef CL_DTYPE_half
#define CL_DTYPE half
#define CL_DTYPE_CHAR h
#define CL_COMPUTE_DTYPE half
#define CL_COMPUTE_DTYPE_CHAR h
#endif

/////////////////////////////////
// GET_VEC_TYPE
/////////////////////////////////
// Note: macro name replacement need twice parser
#define GET_VEC_TYPE(type__, size__) type__##size__
#define VECTORIZED_TYPE(type__, size__) GET_VEC_TYPE(type__, size__)
#define CL_DTYPE4 VECTORIZED_TYPE(CL_DTYPE, 4)
#define CL_COMPUTE_DTYPE4 VECTORIZED_TYPE(CL_COMPUTE_DTYPE, 4)

/////////////////////////////////
// CONVERT_TYPE_TO
/////////////////////////////////
#define _CONVERT_TYPE_TO(value, type) convert_##type(value)
#define CONVERT_TYPE_TO(value, type) _CONVERT_TYPE_TO(value, type)

/////////////////////////////////
// WRITE_IMG_TYPE / READ_IMG_TYPE
/////////////////////////////////
#define _WRITE_IMG_TYPE(type_char, img, pos, value) \
  write_image##type_char(img, pos, value)
#define WRITE_IMG_TYPE(type_char, img, pos, value) \
  _WRITE_IMG_TYPE(type_char, img, pos, value)

#define _READ_IMG_TYPE(type_char, img, sampler, pos) \
  read_image##type_char(img, sampler, pos)
#define READ_IMG_TYPE(type_char, img, sampler, pos) \
  _READ_IMG_TYPE(type_char, img, sampler, pos)

/////////////////////////////////
// activation / activation_type4
/////////////////////////////////
inline CL_DTYPE activation(CL_DTYPE in
#ifdef PRELU
                           ,
                           CL_DTYPE prelu_alpha
#endif
                           ) {
  CL_DTYPE output = in;
#ifdef PRELU
  output = select(prelu_alpha * in, in, in >= (CL_DTYPE)0);
#endif

#ifdef RELU
  output = fmax(in, (CL_DTYPE)0);
#endif

#ifdef RELU6
  output = clamp(in, (CL_DTYPE)0, (CL_DTYPE)6);
#endif

#ifdef LEAKY_RELU
  output =
      select((CL_DTYPE)(LEAKY_RELU_ALPHA)*in, (CL_DTYPE)in, (ushort)(in >= 0));
#endif
  return output;
}

inline CL_DTYPE4 activation_type4(CL_DTYPE4 in
#ifdef PRELU
                                  ,
                                  CL_DTYPE4 prelu_alpha
#endif
                                  ) {
  CL_DTYPE4 output = in;
#ifdef PRELU
  output = select(prelu_alpha * in, in, in >= (CL_DTYPE4)0.0);
#endif

#ifdef RELU
  output = fmax(in, (CL_DTYPE4)0);
#endif

#ifdef RELU6
  in = fmax((CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f), in);
  output = fmin((CL_DTYPE4)(6.0f, 6.0f, 6.0f, 6.0f), in);
#endif

#ifdef LEAKY_RELU
  // note: `(ushort4)(in >= 0)` causes error: invalid conversion
  // between ext-vector type 'ushort4' and 'short
  // __attribute__((ext_vector_type(4)))'
  // thus, using `(ushort4)(in.x >= 0, in.y >= 0, in.z >= 0, in.w >= 0)`
  // instead.
  output = select((CL_DTYPE4)(LEAKY_RELU_ALPHA)*in,
                  (CL_DTYPE4)in,
                  (ushort4)(in.x >= 0, in.y >= 0, in.z >= 0, in.w >= 0));
#endif
  return output;
}
