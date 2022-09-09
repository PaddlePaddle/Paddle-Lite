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

__kernel void conv2d_3x3(__private const int global_size_dim0,
                         __private const int global_size_dim1,
                         __private const int global_size_dim2,
                         __read_only image2d_t input_image,
                         __read_only image2d_t filter,
                         __read_only image2d_t bias,
                         __write_only image2d_t output_image,
                         __private const int stride,
                         __private const int offset,
                         __private const int input_c,
                         __private const int dilation,
                         __private const int input_width,  /* of one block */
                         __private const int input_height, /* of one block */
                         __private const int output_width,
                         __private const int output_height,
                         __private const int output_c,
                         __private const int filter_tensor_c,
                         __private const int filter_width,
                         __private const int filter_height,
                         __private const int group,
                         __private const int input_tensor_c,
                         __read_only image2d_t prelu_alpha) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  if (out_c >= global_size_dim0 || out_w >= global_size_dim1 ||
      out_nh >= global_size_dim2) {
    return;
  }

  int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);

  if (out_c >= global_size_dim0 || out_w >= global_size_dim1 ||
      out_nh >= global_size_dim2) {
    return;
  }

  int2 stride_xy;
  stride_xy.x = stride;
  stride_xy.y = stride;

  int2 ouput_pos_in_one_block;
  ouput_pos_in_one_block.x = out_w;
  ouput_pos_in_one_block.y = out_nh;

  int2 in_pos_in_one_block;
  in_pos_in_one_block.x = ouput_pos_in_one_block.x * stride + offset;
  in_pos_in_one_block.y = ouput_pos_in_one_block.y * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 output =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
  CL_DTYPE4 output = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, output_pos);
#else
  CL_DTYPE4 output = (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f);
#endif
  CL_DTYPE4 zero_dtype4 = (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f);

  CL_DTYPE4 input0, input1, input2, input3, input4, input5, input6, input7,
      input8;
  for (int i = 0; i < 4; i++) {
    int used_input_channel_num =
        (out_c * 4 + i) / (output_c / group) * filter_tensor_c;
    for (int filter_tensor_c_idx = 0; filter_tensor_c_idx < filter_tensor_c;
         ++filter_tensor_c_idx) {
      int input_c = used_input_channel_num + filter_tensor_c_idx;
      int input_block = input_c / 4;
      int2 pos_in = (int2)(input_block * input_width + in_pos_in_one_block.x,
                           in_pos_in_one_block.y);
      input0 = SELECT(
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input_image,
                        SAMPLER,
                        (int2)(pos_in.x - dilation, pos_in.y - dilation)),
          zero_dtype4,
          in_pos_in_one_block.x - dilation < 0 ||
              in_pos_in_one_block.y - dilation < 0 ||
              in_pos_in_one_block.x - dilation >= input_width ||
              in_pos_in_one_block.y - dilation >= input_height);
      input1 = SELECT(READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    input_image,
                                    SAMPLER,
                                    (int2)(pos_in.x, pos_in.y - dilation)),
                      zero_dtype4,
                      in_pos_in_one_block.x < 0 ||
                          in_pos_in_one_block.y - dilation < 0 ||
                          in_pos_in_one_block.x >= input_width ||
                          in_pos_in_one_block.y - dilation >= input_height);
      input2 = SELECT(
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input_image,
                        SAMPLER,
                        (int2)(pos_in.x + dilation, pos_in.y - dilation)),
          zero_dtype4,
          in_pos_in_one_block.x + dilation < 0 ||
              in_pos_in_one_block.y - dilation < 0 ||
              in_pos_in_one_block.x + dilation >= input_width ||
              in_pos_in_one_block.y - dilation >= input_height);

      input3 = SELECT(READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    input_image,
                                    SAMPLER,
                                    (int2)(pos_in.x - dilation, pos_in.y)),
                      zero_dtype4,
                      in_pos_in_one_block.x - dilation < 0 ||
                          in_pos_in_one_block.y < 0 ||
                          in_pos_in_one_block.x - dilation >= input_width ||
                          in_pos_in_one_block.y >= input_height);

      input4 = SELECT(
          READ_IMG_TYPE(
              CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(pos_in.x, pos_in.y)),
          zero_dtype4,
          in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 ||
              in_pos_in_one_block.x >= input_width ||
              in_pos_in_one_block.y >= input_height);
      input5 = SELECT(READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    input_image,
                                    SAMPLER,
                                    (int2)(pos_in.x + dilation, pos_in.y)),
                      zero_dtype4,
                      in_pos_in_one_block.x + dilation < 0 ||
                          in_pos_in_one_block.y < 0 ||
                          in_pos_in_one_block.x + dilation >= input_width ||
                          in_pos_in_one_block.y >= input_height);
      input6 = SELECT(
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input_image,
                        SAMPLER,
                        (int2)(pos_in.x - dilation, pos_in.y + dilation)),
          zero_dtype4,
          in_pos_in_one_block.x - dilation < 0 ||
              in_pos_in_one_block.y + dilation < 0 ||
              in_pos_in_one_block.x - dilation >= input_width ||
              in_pos_in_one_block.y + dilation >= input_height);
      input7 = SELECT(READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    input_image,
                                    SAMPLER,
                                    (int2)(pos_in.x, pos_in.y + dilation)),
                      zero_dtype4,
                      in_pos_in_one_block.x < 0 ||
                          in_pos_in_one_block.y + dilation < 0 ||
                          in_pos_in_one_block.x >= input_width ||
                          in_pos_in_one_block.y + dilation >= input_height);
      input8 = SELECT(
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input_image,
                        SAMPLER,
                        (int2)(pos_in.x + dilation, pos_in.y + dilation)),
          zero_dtype4,
          in_pos_in_one_block.x + dilation < 0 ||
              in_pos_in_one_block.y + dilation < 0 ||
              in_pos_in_one_block.x + dilation >= input_width ||
              in_pos_in_one_block.y + dilation >= input_height);

      CL_DTYPE tmp_out = 0;
      for (int j = 0; j < 9; j++) {
        int2 pos_of_weight;
        pos_of_weight.x = (filter_tensor_c_idx / 4) * 3 + j % 3;
        pos_of_weight.y = out_c * 4 * 3 + i * 3 + j / 3;
        CL_DTYPE4 weight =
            READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, pos_of_weight);

        int filter_tensor_c_idx_offset = filter_tensor_c_idx % 4;
        CL_DTYPE f_value = 0;
        f_value = (filter_tensor_c_idx_offset == 0) ? weight.x : f_value;
        f_value = (filter_tensor_c_idx_offset == 1) ? weight.y : f_value;
        f_value = (filter_tensor_c_idx_offset == 2) ? weight.z : f_value;
        f_value = (filter_tensor_c_idx_offset == 3) ? weight.w : f_value;

        int input_c_offset = input_c % 4;
        CL_DTYPE input_value = 0;
        if (j == 0) {
          input_value = (input_c_offset == 0) ? input0.x : input_value;
          input_value = (input_c_offset == 1) ? input0.y : input_value;
          input_value = (input_c_offset == 2) ? input0.z : input_value;
          input_value = (input_c_offset == 3) ? input0.w : input_value;
        } else if (j == 1) {
          input_value = (input_c_offset == 0) ? input1.x : input_value;
          input_value = (input_c_offset == 1) ? input1.y : input_value;
          input_value = (input_c_offset == 2) ? input1.z : input_value;
          input_value = (input_c_offset == 3) ? input1.w : input_value;
        } else if (j == 2) {
          input_value = (input_c_offset == 0) ? input2.x : input_value;
          input_value = (input_c_offset == 1) ? input2.y : input_value;
          input_value = (input_c_offset == 2) ? input2.z : input_value;
          input_value = (input_c_offset == 3) ? input2.w : input_value;
        } else if (j == 3) {
          input_value = (input_c_offset == 0) ? input3.x : input_value;
          input_value = (input_c_offset == 1) ? input3.y : input_value;
          input_value = (input_c_offset == 2) ? input3.z : input_value;
          input_value = (input_c_offset == 3) ? input3.w : input_value;
        } else if (j == 4) {
          input_value = (input_c_offset == 0) ? input4.x : input_value;
          input_value = (input_c_offset == 1) ? input4.y : input_value;
          input_value = (input_c_offset == 2) ? input4.z : input_value;
          input_value = (input_c_offset == 3) ? input4.w : input_value;
        } else if (j == 5) {
          input_value = (input_c_offset == 0) ? input5.x : input_value;
          input_value = (input_c_offset == 1) ? input5.y : input_value;
          input_value = (input_c_offset == 2) ? input5.z : input_value;
          input_value = (input_c_offset == 3) ? input5.w : input_value;
        } else if (j == 6) {
          input_value = (input_c_offset == 0) ? input6.x : input_value;
          input_value = (input_c_offset == 1) ? input6.y : input_value;
          input_value = (input_c_offset == 2) ? input6.z : input_value;
          input_value = (input_c_offset == 3) ? input6.w : input_value;
        } else if (j == 7) {
          input_value = (input_c_offset == 0) ? input7.x : input_value;
          input_value = (input_c_offset == 1) ? input7.y : input_value;
          input_value = (input_c_offset == 2) ? input7.z : input_value;
          input_value = (input_c_offset == 3) ? input7.w : input_value;
        } else if (j == 8) {
          input_value = (input_c_offset == 0) ? input8.x : input_value;
          input_value = (input_c_offset == 1) ? input8.y : input_value;
          input_value = (input_c_offset == 2) ? input8.z : input_value;
          input_value = (input_c_offset == 3) ? input8.w : input_value;
        }

        tmp_out += f_value * input_value;
      }
      output.x = (i == 0) ? output.x + tmp_out : output.x;
      output.y = (i == 1) ? output.y + tmp_out : output.y;
      output.z = (i == 2) ? output.z + tmp_out : output.z;
      output.w = (i == 3) ? output.w + tmp_out : output.w;
    }
  }

  CL_DTYPE4 alpha0;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      prelu_alpha,
      SAMPLER,
      (int2)(out_c * global_size_dim1 + out_w, out_nh % output_height));
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
//}
#endif
  output = activation_type4(output, alpha0);

#ifdef SCALE_ACTIVATION
  output = fuse_scale(output, 1.f, 0.f, 0.f);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);
}
