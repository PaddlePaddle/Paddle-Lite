/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/fpga/KD/fpga_cv.hpp"

using paddle::zynqmp::float16;

void fpga_resize(float* input,
                 int input_width,
                 int input_height,
                 int input_channel,
                 uint8_t* output,
                 int output_width,
                 int output_height) {
  paddle::zynqmp::InplaceArgs inplace_args = {
      .relu_enable = 0, .power_enable = 0,
  };
  paddle::zynqmp::config_inplace(inplace_args);

  paddle::zynqmp::ImageInputArgs input_args = {nullptr};
  input_args.address = nullptr;
  input_args.scale_address = nullptr;

  float16* input_image_address =
      reinterpret_cast<float16*>(paddle::zynqmp::fpga_malloc(
          input_width * input_height * input_channel * sizeof(float16)));
  int index = 0;

  for (int i = 0; i < input_width * input_height * input_channel; i++) {
    input_image_address[i] = float16(1.0 * input[i]);
  }

  paddle::zynqmp::ResizeArgs resize_args = {0};

  resize_args.input_width = input_width;
  resize_args.input_height = input_height;
  resize_args.image_channel = input_channel;
  resize_args.output_width = output_width;
  resize_args.output_height = output_height;
  float height_ratio = static_cast<float>(input_height) /
                       static_cast<float>(resize_args.output_height);
  float width_ratio = static_cast<float>(input_width) /
                      static_cast<float>(resize_args.output_width);
  resize_args.height_ratio = *reinterpret_cast<uint32_t*>(&height_ratio);
  resize_args.width_ratio = *reinterpret_cast<uint32_t*>(&width_ratio);

  int output_size =
      resize_args.output_width * resize_args.output_height * input_channel;
  float16* fpga_output = reinterpret_cast<float16*>(
      paddle::zynqmp::fpga_malloc(output_size * sizeof(float16)));
  resize_args.input_image_address = input_image_address;
  resize_args.output_image_address = fpga_output;

  memset(fpga_output, 0, output_size * sizeof(float16));
  paddle::zynqmp::fpga_flush(
      input_image_address,
      input_width * input_height * input_channel * sizeof(float16));
  paddle::zynqmp::fpga_flush(resize_args.output_image_address,
                             output_size * sizeof(float16));
  int ret = paddle::zynqmp::compute_fpga_resize(resize_args);
  if (ret == 0) {
    paddle::zynqmp::fpga_invalidate(resize_args.output_image_address,
                                    output_size * sizeof(float16));
  }

  for (int i = 0; i < output_size; i++) {
    output[i] = fpga_output[i];
  }
}
