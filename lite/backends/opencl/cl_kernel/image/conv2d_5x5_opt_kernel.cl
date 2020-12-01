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

// opt version of conv5x5
__kernel void conv2d_5x5_opt(__private const int item_ch,
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
                             __private const int out_h) {
  // filter
  const int filter_w = 5;
  const int filter_h = 5;

  // item_id
  const int item_ch_id = get_global_id(0);
  const int item_w_id = get_global_id(1);
  const int item_h_id = get_global_id(2);

  // out_width_id_per_blk and out_batch_id
  int out_w_base_id = item_ch_id * out_w;
  int out_w_id0 = item_w_id;
  int out_w_id1 = out_w_id0 + item_w;
  int out_w_id2 = out_w_id1 + item_w;
  int out_w_id3 = out_w_id2 + item_w;
  int out_w_id4 = out_w_id3 + item_w;

  // in_width_id_per_blk and in_height_id_per_batch
  int in_h_id = (item_h_id % out_h) * stride - pad;
  int in_w_id0 = item_w_id * stride - pad;
  int in_w_id1 = in_w_id0 + item_w * stride;
  int in_w_id2 = in_w_id1 + item_w * stride;
  int in_w_id3 = in_w_id2 + item_w * stride;
  int in_w_id4 = in_w_id3 + item_w * stride;

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
  CL_DTYPE4 filter_trans[4] = {0.0f};
  CL_DTYPE4 input[5] = {0.0f};

  int filter_h_val0 = item_ch_id * 4 * filter_h;
  int filter_h_val1 = filter_h_val0 + filter_h;
  int filter_h_val2 = filter_h_val1 + filter_h;
  int filter_h_val3 = filter_h_val2 + filter_h;

  for (int ch = 0; ch < (in_ch + 3) / 4; ch++) {
    int ch_surplus = (ch + 1) * 4 - in_ch > 0 ? (ch + 1) * 4 - in_ch : 0;

    const int in_w_base_id = mul24(ch, in_w);

    int filter_w_val = ch * filter_w;

    for (int h = 0; h < filter_h; h++) {
      int in_h_val =
          select(in_h_id + h, -1, (in_h_id + h < 0 || in_h_id + h >= in_h));

      for (int w = 0; w < filter_w; w++) {
        int in_w_val0 = select(in_w_base_id + in_w_id0 + w,
                               -1,
                               (in_w_id0 + w < 0 || in_w_id0 + w >= in_w));
        int in_w_val1 = select(in_w_base_id + in_w_id1 + w,
                               -1,
                               (in_w_id1 + w < 0 || in_w_id1 + w >= in_w));
        int in_w_val2 = select(in_w_base_id + in_w_id2 + w,
                               -1,
                               (in_w_id2 + w < 0 || in_w_id2 + w >= in_w));
        int in_w_val3 = select(in_w_base_id + in_w_id3 + w,
                               -1,
                               (in_w_id3 + w < 0 || in_w_id3 + w >= in_w));
        int in_w_val4 = select(in_w_base_id + in_w_id4 + w,
                               -1,
                               (in_w_id4 + w < 0 || in_w_id4 + w >= in_w));

        filter[0] =
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          filter_image,
                          SAMPLER,
                          (int2)(filter_w_val + w,
                                 filter_h_val0 + h));  // in_ch:0-3,out_ch:0
        filter[1] =
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          filter_image,
                          SAMPLER,
                          (int2)(filter_w_val + w,
                                 filter_h_val1 + h));  // in_ch:0-3,out_ch:1
        filter[2] =
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          filter_image,
                          SAMPLER,
                          (int2)(filter_w_val + w,
                                 filter_h_val2 + h));  // in_ch:0-3,out_ch:2
        filter[3] =
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          filter_image,
                          SAMPLER,
                          (int2)(filter_w_val + w,
                                 filter_h_val3 + h));  // in_ch:0-3,out_ch:3

        filter_trans[0] = (CL_DTYPE4)(filter[0].x,
                                      filter[1].x,
                                      filter[2].x,
                                      filter[3].x);  // in_ch:0,out_ch:0-3
        filter_trans[1] = (CL_DTYPE4)(filter[0].y,
                                      filter[1].y,
                                      filter[2].y,
                                      filter[3].y);  // in_ch:1,out_ch:0-3
        filter_trans[2] = (CL_DTYPE4)(filter[0].z,
                                      filter[1].z,
                                      filter[2].z,
                                      filter[3].z);  // in_ch:2,out_ch:0-3
        filter_trans[3] = (CL_DTYPE4)(filter[0].w,
                                      filter[1].w,
                                      filter[2].w,
                                      filter[3].w);  // in_ch:3,out_ch:0-3

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

        output[0] = mad(input[0].x, filter_trans[0], output[0]);
        output[1] = mad(input[1].x, filter_trans[0], output[1]);
        output[2] = mad(input[2].x, filter_trans[0], output[2]);
        output[3] = mad(input[3].x, filter_trans[0], output[3]);
        output[4] = mad(input[4].x, filter_trans[0], output[4]);

        if (ch_surplus < 3) {
          output[0] = mad(input[0].y, filter_trans[1], output[0]);
          output[1] = mad(input[1].y, filter_trans[1], output[1]);
          output[2] = mad(input[2].y, filter_trans[1], output[2]);
          output[3] = mad(input[3].y, filter_trans[1], output[3]);
          output[4] = mad(input[4].y, filter_trans[1], output[4]);
        }
        if (ch_surplus < 2) {
          output[0] = mad(input[0].z, filter_trans[2], output[0]);
          output[1] = mad(input[1].z, filter_trans[2], output[1]);
          output[2] = mad(input[2].z, filter_trans[2], output[2]);
          output[3] = mad(input[3].z, filter_trans[2], output[3]);
          output[4] = mad(input[4].z, filter_trans[2], output[4]);
        }
        if (ch_surplus < 1) {
          output[0] = mad(input[0].w, filter_trans[3], output[0]);
          output[1] = mad(input[1].w, filter_trans[3], output[1]);
          output[2] = mad(input[2].w, filter_trans[3], output[2]);
          output[3] = mad(input[3].w, filter_trans[3], output[3]);
          output[4] = mad(input[4].w, filter_trans[3], output[4]);
        }
      }
    }
  }

  output[0] = activation_type4(output[0]);
  output[1] = activation_type4(output[1]);
  output[2] = activation_type4(output[2]);
  output[3] = activation_type4(output[3]);
  output[4] = activation_type4(output[4]);

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
// support batch > 1
__kernel void conv2d_5x5_multi_batch(__private const int item_ch,
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
                                     __private const int out_h) {
  // filter
  const int filter_w = 5;
  const int filter_h = 5;

  // item_id
  const int item_ch_id = get_global_id(0);
  const int item_w_id = get_global_id(1);
  const int item_h_id = get_global_id(2);

  // out_width_id_per_blk and out_batch_id
  int out_batch_id = item_h_id / in_h;
  int out_w_base_id = item_ch_id * out_w;
  int out_w_id0 = item_w_id;
  int out_w_id1 = out_w_id0 + item_w;
  int out_w_id2 = out_w_id1 + item_w;
  int out_w_id3 = out_w_id2 + item_w;
  int out_w_id4 = out_w_id3 + item_w;

  // in_width_id_per_blk and in_height_id_per_batch
  int in_h_id = (item_h_id % out_h) * stride - pad;
  int in_w_id0 = item_w_id * stride - pad;
  int in_w_id1 = in_w_id0 + item_w * stride;
  int in_w_id2 = in_w_id1 + item_w * stride;
  int in_w_id3 = in_w_id2 + item_w * stride;
  int in_w_id4 = in_w_id3 + item_w * stride;

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
  CL_DTYPE4 filter_trans[4] = {0.0f};
  CL_DTYPE4 input[5] = {0.0f};

  int filter_h_val0 = item_ch_id * 4 * filter_h;
  int filter_h_val1 = filter_h_val0 + filter_h;
  int filter_h_val2 = filter_h_val1 + filter_h;
  int filter_h_val3 = filter_h_val2 + filter_h;

  for (int ch = 0; ch < (in_ch + 3) / 4; ch++) {
    int ch_surplus = (ch + 1) * 4 - in_ch > 0 ? (ch + 1) * 4 - in_ch : 0;

    const int in_w_base_id = mul24(ch, in_w);

    int filter_w_val = ch * filter_w;

    for (int h = 0; h < filter_h; h++) {
      int in_h_val = select(
          out_batch_id * in_h + in_h_id + h,
          -1,
          (out_batch_id * in_h + in_h_id + h < out_batch_id * in_h ||
           out_batch_id * in_h + in_h_id + h >= (out_batch_id + 1) * in_h));

      for (int w = 0; w < filter_w; w++) {
        int in_w_val0 = select(in_w_base_id + in_w_id0 + w,
                               -1,
                               (in_w_id0 + w < 0 || in_w_id0 + w >= in_w));
        int in_w_val1 = select(in_w_base_id + in_w_id1 + w,
                               -1,
                               (in_w_id1 + w < 0 || in_w_id1 + w >= in_w));
        int in_w_val2 = select(in_w_base_id + in_w_id2 + w,
                               -1,
                               (in_w_id2 + w < 0 || in_w_id2 + w >= in_w));
        int in_w_val3 = select(in_w_base_id + in_w_id3 + w,
                               -1,
                               (in_w_id3 + w < 0 || in_w_id3 + w >= in_w));
        int in_w_val4 = select(in_w_base_id + in_w_id4 + w,
                               -1,
                               (in_w_id4 + w < 0 || in_w_id4 + w >= in_w));

        filter[0] =
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          filter_image,
                          SAMPLER,
                          (int2)(filter_w_val + w,
                                 filter_h_val0 + h));  // in_ch:0-3,out_ch:0
        filter[1] =
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          filter_image,
                          SAMPLER,
                          (int2)(filter_w_val + w,
                                 filter_h_val1 + h));  // in_ch:0-3,out_ch:1
        filter[2] =
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          filter_image,
                          SAMPLER,
                          (int2)(filter_w_val + w,
                                 filter_h_val2 + h));  // in_ch:0-3,out_ch:2
        filter[3] =
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          filter_image,
                          SAMPLER,
                          (int2)(filter_w_val + w,
                                 filter_h_val3 + h));  // in_ch:0-3,out_ch:3

        filter_trans[0] = (CL_DTYPE4)(filter[0].x,
                                      filter[1].x,
                                      filter[2].x,
                                      filter[3].x);  // in_ch:0,out_ch:0-3
        filter_trans[1] = (CL_DTYPE4)(filter[0].y,
                                      filter[1].y,
                                      filter[2].y,
                                      filter[3].y);  // in_ch:1,out_ch:0-3
        filter_trans[2] = (CL_DTYPE4)(filter[0].z,
                                      filter[1].z,
                                      filter[2].z,
                                      filter[3].z);  // in_ch:2,out_ch:0-3
        filter_trans[3] = (CL_DTYPE4)(filter[0].w,
                                      filter[1].w,
                                      filter[2].w,
                                      filter[3].w);  // in_ch:3,out_ch:0-3

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

        output[0] = mad(input[0].x, filter_trans[0], output[0]);
        output[1] = mad(input[1].x, filter_trans[0], output[1]);
        output[2] = mad(input[2].x, filter_trans[0], output[2]);
        output[3] = mad(input[3].x, filter_trans[0], output[3]);
        output[4] = mad(input[4].x, filter_trans[0], output[4]);

        if (ch_surplus < 3) {
          output[0] = mad(input[0].y, filter_trans[1], output[0]);
          output[1] = mad(input[1].y, filter_trans[1], output[1]);
          output[2] = mad(input[2].y, filter_trans[1], output[2]);
          output[3] = mad(input[3].y, filter_trans[1], output[3]);
          output[4] = mad(input[4].y, filter_trans[1], output[4]);
        }
        if (ch_surplus < 2) {
          output[0] = mad(input[0].z, filter_trans[2], output[0]);
          output[1] = mad(input[1].z, filter_trans[2], output[1]);
          output[2] = mad(input[2].z, filter_trans[2], output[2]);
          output[3] = mad(input[3].z, filter_trans[2], output[3]);
          output[4] = mad(input[4].z, filter_trans[2], output[4]);
        }
        if (ch_surplus < 1) {
          output[0] = mad(input[0].w, filter_trans[3], output[0]);
          output[1] = mad(input[1].w, filter_trans[3], output[1]);
          output[2] = mad(input[2].w, filter_trans[3], output[2]);
          output[3] = mad(input[3].w, filter_trans[3], output[3]);
          output[4] = mad(input[4].w, filter_trans[3], output[4]);
        }
      }
    }
  }

  output[0] = activation_type4(output[0]);
  output[1] = activation_type4(output[1]);
  output[2] = activation_type4(output[2]);
  output[3] = activation_type4(output[3]);
  output[4] = activation_type4(output[4]);

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
