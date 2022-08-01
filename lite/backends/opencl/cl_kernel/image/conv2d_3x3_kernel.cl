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

// support batch > 1
__kernel void conv2d_3x3_multi_batch(__private const int item_ch,
                                     __private const int item_w,
                                     __private const int item_h,
                                     __read_only image2d_t input_image,
                                     __read_only image2d_t filter_image,
                                     __read_only image2d_t bias,
                                     __write_only image2d_t output_image,
                                     __private const int stride,
                                     __private const int pad,
                                     __private const int dilation,
                                     __private const int batch,
                                     __private const int in_ch,
                                     __private const int in_w,
                                     __private const int in_h,
                                     __private const int out_w,
                                     __private const int out_h,
                                     __read_only image2d_t prelu_alpha) {
  // item_id
  const int item_ch_id = get_global_id(0);
  const int item_w_id = get_global_id(1);
  const int item_h_id = get_global_id(2);
  if (item_ch_id >= item_ch || item_w_id >= item_w || item_h_id >= item_h) {
    return;
  }

  // out_width_id_per_blk
  int out_batch_id = item_h_id / out_h;
  int out_w_base_id = mul24(item_ch_id, out_w);
  int out_w_id0 = item_w_id;
  int out_w_id1 = out_w_id0 + item_w;
  int out_w_id2 = out_w_id1 + item_w;
  int out_w_id3 = out_w_id2 + item_w;
  int out_w_id4 = out_w_id3 + item_w;

  // in_width_id_per_blk and in_height_id_per_batch
  int in_h_id = mad24((item_h_id % out_h), stride, (-pad));
  int in_w_id0 = mad24(item_w_id, stride, (-pad));
  int in_w_id1 = mad24(item_w, stride, in_w_id0);
  int in_w_id2 = mad24(item_w, stride, in_w_id1);
  int in_w_id3 = mad24(item_w, stride, in_w_id2);
  int in_w_id4 = mad24(item_w, stride, in_w_id3);

#ifdef BIASE_CH

  CL_DTYPE4 output[5];
  output[0] =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(item_ch_id, 0));
  output[1] = output[0];
  output[2] = output[0];
  output[3] = output[0];
  output[4] = output[0];

#elif defined(BIASE_ELE)

  CL_DTYPE4 output[5];
  output[0] = READ_IMG_TYPE(CL_DTYPE_CHAR,
                            bias,
                            SAMPLER,
                            (int2)(out_w_base_id + out_w_id0, item_h_id));
  if (out_w_id1 < out_w) {
    output[1] = READ_IMG_TYPE(CL_DTYPE_CHAR,
                              bias,
                              SAMPLER,
                              (int2)(out_w_base_id + out_w_id1, item_h_id));
  }
  if (out_w_id2 < out_w) {
    output[2] = READ_IMG_TYPE(CL_DTYPE_CHAR,
                              bias,
                              SAMPLER,
                              (int2)(out_w_base_id + out_w_id2, item_h_id));
  }
  if (out_w_id3 < out_w) {
    output[3] = READ_IMG_TYPE(CL_DTYPE_CHAR,
                              bias,
                              SAMPLER,
                              (int2)(out_w_base_id + out_w_id3, item_h_id));
  }
  if (out_w_id4 < out_w) {
    output[4] = READ_IMG_TYPE(CL_DTYPE_CHAR,
                              bias,
                              SAMPLER,
                              (int2)(out_w_base_id + out_w_id4, item_h_id));
  }
#else
  CL_DTYPE4 output[5] = {0.0f};
#endif

  CL_DTYPE4 filter[4] = {0.0f};
  CL_DTYPE4 input[5] = {0.0f};

  for (int ch = 0; ch < ((in_ch + 3) >> 2); ch++) {
    int ch_surplus = ((ch + 1) << 2) - in_ch > 0 ? ((ch + 1) << 2) - in_ch : 0;

    const int in_w_base_id = mul24(ch, in_w);

    int filter_w_val = ch << 2;
    int filter_h_val = mul24(item_ch_id, 9);

    for (int h = 0; h < 3; h++) {
      int in_h_val = select(mad24(out_batch_id, in_h, (in_h_id + h)),
                            -1,
                            (mad24(out_batch_id, in_h, (in_h_id + h)) <
                                 mul24(out_batch_id, in_h) ||
                             mad24(out_batch_id, in_h, (in_h_id + h)) >=
                                 mul24((out_batch_id + 1), in_h)));

      for (int w = 0; w < 3; w++) {
        int in_w_val0 = select(in_w_base_id + in_w_id0 + w,
                               -1,
                               (in_w_id0 + w < 0 | in_w_id0 + w >= in_w));
        int in_w_val1 = select(in_w_base_id + in_w_id1 + w,
                               -1,
                               (in_w_id1 + w < 0 | in_w_id1 + w >= in_w));
        int in_w_val2 = select(in_w_base_id + in_w_id2 + w,
                               -1,
                               (in_w_id2 + w < 0 | in_w_id2 + w >= in_w));
        int in_w_val3 = select(in_w_base_id + in_w_id3 + w,
                               -1,
                               (in_w_id3 + w < 0 | in_w_id3 + w >= in_w));
        int in_w_val4 = select(in_w_base_id + in_w_id4 + w,
                               -1,
                               (in_w_id4 + w < 0 | in_w_id4 + w >= in_w));

        filter[0] = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                  filter_image,
                                  SAMPLER,
                                  (int2)(filter_w_val, filter_h_val));
        filter[1] = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                  filter_image,
                                  SAMPLER,
                                  (int2)(filter_w_val + 1, filter_h_val));
        filter[2] = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                  filter_image,
                                  SAMPLER,
                                  (int2)(filter_w_val + 2, filter_h_val));
        filter[3] = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                  filter_image,
                                  SAMPLER,
                                  (int2)(filter_w_val + 3, filter_h_val++));

        input[0] = READ_IMG_TYPE(
            CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(in_w_val0, in_h_val));
        input[1] = READ_IMG_TYPE(
            CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(in_w_val1, in_h_val));
        input[2] = READ_IMG_TYPE(
            CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(in_w_val2, in_h_val));
        input[3] = READ_IMG_TYPE(
            CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(in_w_val3, in_h_val));
        input[4] = READ_IMG_TYPE(
            CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(in_w_val4, in_h_val));

        output[0] = mad(input[0].x, filter[0], output[0]);
        output[1] = mad(input[1].x, filter[0], output[1]);
        output[2] = mad(input[2].x, filter[0], output[2]);
        output[3] = mad(input[3].x, filter[0], output[3]);
        output[4] = mad(input[4].x, filter[0], output[4]);

        if (ch_surplus < 3) {
          output[0] = mad(input[0].y, filter[1], output[0]);
          output[1] = mad(input[1].y, filter[1], output[1]);
          output[2] = mad(input[2].y, filter[1], output[2]);
          output[3] = mad(input[3].y, filter[1], output[3]);
          output[4] = mad(input[4].y, filter[1], output[4]);
        }
        if (ch_surplus < 2) {
          output[0] = mad(input[0].z, filter[2], output[0]);
          output[1] = mad(input[1].z, filter[2], output[1]);
          output[2] = mad(input[2].z, filter[2], output[2]);
          output[3] = mad(input[3].z, filter[2], output[3]);
          output[4] = mad(input[4].z, filter[2], output[4]);
        }
        if (ch_surplus < 1) {
          output[0] = mad(input[0].w, filter[3], output[0]);
          output[1] = mad(input[1].w, filter[3], output[1]);
          output[2] = mad(input[2].w, filter[3], output[2]);
          output[3] = mad(input[3].w, filter[3], output[3]);
          output[4] = mad(input[4].w, filter[3], output[4]);
        }
      }
    }
  }

  CL_DTYPE4 alpha[5];
#ifdef PRELU_CH  //{
  alpha[0] =
      READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(item_ch_id, 0));
  alpha[1] = alpha[0];
  alpha[2] = alpha[0];
  alpha[3] = alpha[0];
  alpha[4] = alpha[0];
//}
#elif defined(PRELU_ELE)  //{
  alpha[0] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prelu_alpha,
                    SAMPLER,
                    (int2)(out_w_base_id + out_w_id0, item_h_id % out_h));
  if (out_w_id1 < out_w) {
    alpha[1] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prelu_alpha,
                      SAMPLER,
                      (int2)(out_w_base_id + out_w_id1, item_h_id % out_h));
  }
  if (out_w_id2 < out_w) {
    alpha[2] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prelu_alpha,
                      SAMPLER,
                      (int2)(out_w_base_id + out_w_id2, item_h_id % out_h));
  }
  if (out_w_id3 < out_w) {
    alpha[3] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prelu_alpha,
                      SAMPLER,
                      (int2)(out_w_base_id + out_w_id3, item_h_id % out_h));
  }
  if (out_w_id4 < out_w) {
    alpha[4] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prelu_alpha,
                      SAMPLER,
                      (int2)(out_w_base_id + out_w_id4, item_h_id % out_h));
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha[0] = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha[0].y = alpha[0].x;
  alpha[0].z = alpha[0].x;
  alpha[0].w = alpha[0].x;
  alpha[1] = alpha[0];
  alpha[2] = alpha[0];
  alpha[3] = alpha[0];
  alpha[4] = alpha[0];
//}
#endif
  output[0] = activation_type4(output[0], alpha[0]);
  output[1] = activation_type4(output[1], alpha[1]);
  output[2] = activation_type4(output[2], alpha[2]);
  output[3] = activation_type4(output[3], alpha[3]);
  output[4] = activation_type4(output[4], alpha[4]);

#ifdef SCALE_ACTIVATION
  output[0] = fuse_scale(output[0], 1.f, 0.f, 0.f);
  output[1] = fuse_scale(output[1], 1.f, 0.f, 0.f);
  output[2] = fuse_scale(output[2], 1.f, 0.f, 0.f);
  output[3] = fuse_scale(output[3], 1.f, 0.f, 0.f);
  output[4] = fuse_scale(output[4], 1.f, 0.f, 0.f);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(out_w_base_id + out_w_id0, item_h_id),
                 output[0]);
  if (out_w_id1 < out_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_w_base_id + out_w_id1, item_h_id),
                   output[1]);
  }
  if (out_w_id2 < out_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_w_base_id + out_w_id2, item_h_id),
                   output[2]);
  }
  if (out_w_id3 < out_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_w_base_id + out_w_id3, item_h_id),
                   output[3]);
  }
  if (out_w_id4 < out_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_w_base_id + out_w_id4, item_h_id),
                   output[4]);
  }
}