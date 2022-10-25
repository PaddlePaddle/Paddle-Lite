/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void conv2d_nxn_opt_mali(__private const int item_ch,
                                  __private const int item_w,
                                  __private const int item_h,
                                  __read_only image2d_t input_image,
                                  __global CL_DTYPE4 *filter_buf,
                                  __global CL_DTYPE4 *bias_buf,
                                  __private const int filter_w,
                                  __private const int filter_h,
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
  const int item_w_id = get_global_id(1) << 1;
  const int item_nh_id = get_global_id(2);
  if (get_global_id(0) >= item_ch || get_global_id(1) >= item_w ||
      get_global_id(2) >= item_h) {
    return;
  }

  // out_width_id_per_blk
  int out_w_base_id = mul24(item_ch_id, out_w);
  int out_w_id0 = item_w_id;
  int out_w_id1 = out_w_id0 + 1;
  int filter_w_h = filter_w * filter_h;

  // in_width_id_per_blk and in_height_id_per_batch
  int n = item_nh_id / out_h;
  int in_h_id = mad24((item_nh_id % out_h), stride, (-pad));
  int in_w_id0 = mad24(item_w_id, stride, (-pad));
  int in_w_id1 = mad24(item_w_id + 1, stride, (-pad));

#ifdef BIASE_CH
  CL_DTYPE4 output[2];
  output[0] = (bias_buf + item_ch_id)[0];
  output[1] = output[0];
#else
  CL_DTYPE4 output[2] = {0.0f};
#endif

  CL_DTYPE4 input[2] = {0.0f};

  for (int ch = 0; ch < ((in_ch + 3) >> 2); ch++) {
    const int in_w_base_id = mul24(ch, in_w);

    int filter_w_val = ch << 2;
    int filter_h_val = mul24(item_ch_id, filter_w_h);
    __global CL_DTYPE4 *filter_ptr =
        filter_buf + ((in_ch + 3) >> 2) * 4 * filter_h_val + filter_w_val;

    for (int h = 0; h < filter_h; h++) {
      int in_h_val = select(
          n * in_h + in_h_id + h, -1, (in_h_id + h < 0 | in_h_id + h >= in_h));

      for (int w = 0; w < filter_w; w++) {
        int in_w_val0 = select(in_w_base_id + in_w_id0 + w,
                               -1,
                               (in_w_id0 + w < 0 | in_w_id0 + w >= in_w));
        int in_w_val1 = select(in_w_base_id + in_w_id1 + w,
                               -1,
                               (in_w_id1 + w < 0 | in_w_id1 + w >= in_w));

        input[0] = READ_IMG_TYPE(
            CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(in_w_val0, in_h_val));
        input[1] = READ_IMG_TYPE(
            CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(in_w_val1, in_h_val));

        output[0] = mad(input[0].x, filter_ptr[0], output[0]);
        output[1] = mad(input[1].x, filter_ptr[0], output[1]);

        output[0] = mad(input[0].y, filter_ptr[1], output[0]);
        output[1] = mad(input[1].y, filter_ptr[1], output[1]);

        output[0] = mad(input[0].z, filter_ptr[2], output[0]);
        output[1] = mad(input[1].z, filter_ptr[2], output[1]);

        output[0] = mad(input[0].w, filter_ptr[3], output[0]);
        output[1] = mad(input[1].w, filter_ptr[3], output[1]);

        filter_ptr += ((in_ch + 3) >> 2) * 4;
      }
    }
  }
  CL_DTYPE4 alpha[2];
#ifdef PRELU_CH
  alpha[0] =
      READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(item_ch_id, 0));
  alpha[1] = alpha[0];
#elif defined(PRELU_ELE)
  alpha[0] = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_w_base_id + out_w_id0, item_nh_id));
  if (out_w_id1 < out_w) {
    alpha[1] = READ_IMG_TYPE(CL_DTYPE_CHAR,
                             prelu_alpha,
                             SAMPLER,
                             (int2)(out_w_base_id + out_w_id1, item_nh_id));
  }
#elif defined(PRELU_ALL)
  alpha[0] = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha[0].y = alpha[0].x;
  alpha[0].z = alpha[0].x;
  alpha[0].w = alpha[0].x;
  alpha[1] = alpha[0];
#endif
  output[0] = activation_type4(output[0], alpha[0]);
  output[1] = activation_type4(output[1], alpha[1]);

#ifdef SCALE_ACTIVATION
  output[0] = fuse_scale(output[0], 1.f, 0.f, 0.f);
  output[1] = fuse_scale(output[1], 1.f, 0.f, 0.f);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(out_w_base_id + out_w_id0, item_nh_id),
                 output[0]);
  if (out_w_id1 < out_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_w_base_id + out_w_id1, item_nh_id),
                   output[1]);
  }
}
