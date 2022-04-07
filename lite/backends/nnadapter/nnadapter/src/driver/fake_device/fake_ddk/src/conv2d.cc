// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "conv2d.h"  // NOLINT
#include <math.h>
#include <algorithm>
#include <vector>
#include "logging.h"  // NOLINT
#include "utility.h"  // NOLINT

namespace fake_ddk {

int conv2d(Tensor* input_tensor,
           Tensor* filter_tensor,
           Tensor* bias_tensor,
           Tensor* output_tensor,
           Conv2DAttr* conv2d_attr) {
  if (!input_tensor || !filter_tensor || !output_tensor || !conv2d_attr) {
    return StatusType::FAILURE;
  }
  auto& input_shape = input_tensor->attr.shape;
  auto& filter_shape = filter_tensor->attr.shape;
  auto input_rank = input_shape.size();
  auto filter_rank = filter_shape.size();
  if (input_rank != 4 || filter_rank != 4) {
    return StatusType::FAILURE;
  }
  auto batch_size = input_shape[0];
  auto input_channel_size = input_shape[1];
  auto input_height = input_shape[2];
  auto input_width = input_shape[3];
  auto output_channel_size = filter_shape[0];
  auto kernel_height = filter_shape[2];
  auto kernel_width = filter_shape[3];
  auto pad_height_top = conv2d_attr->pad[0];
  auto pad_height_bottom = conv2d_attr->pad[1];
  auto pad_width_left = conv2d_attr->pad[2];
  auto pad_width_right = conv2d_attr->pad[3];
  auto stride_height = conv2d_attr->stride[0];
  auto stride_width = conv2d_attr->stride[1];
  auto dilation_height = conv2d_attr->dilation[0];
  auto dilation_width = conv2d_attr->dilation[1];
  auto group = conv2d_attr->group;
  auto fuse_type = conv2d_attr->fuse_type;
  auto output_height = (input_height + (pad_height_top + pad_height_bottom) -
                        dilation_height * (kernel_height - 1) - 1) /
                           stride_height +
                       1;
  auto output_width = (input_width + (pad_width_left + pad_width_right) -
                       dilation_width * (kernel_width - 1) - 1) /
                          stride_width +
                      1;
  auto output_channel_group = output_channel_size / group;
  auto input_channel_group = input_channel_size / group;
  // Dequantize input data if needs
  auto input_precision = input_tensor->attr.precision;
  auto input_count = shape_production(input_shape);
  auto& input_quant_params = input_tensor->attr.quant_params;
  auto input_buffer = get_tensor_buffer_address(input_tensor);
  auto input_data = reinterpret_cast<float*>(input_buffer);
  std::vector<float> dequantized_input_data;
  if (input_precision == PrecisionType::QUANT_INT8_SYMM_PER_LAYER) {
    dequantized_input_data.resize(input_count);
    input_data = dequantized_input_data.data();
    dequantize(reinterpret_cast<int8_t*>(input_buffer),
               input_shape,
               input_quant_params.scales[0],
               input_data);
  } else if (input_precision == PrecisionType::QUANT_UINT8_ASYMM_PER_LAYER) {
    dequantized_input_data.resize(input_count);
    input_data = dequantized_input_data.data();
    dequantize(reinterpret_cast<uint8_t*>(input_buffer),
               input_shape,
               input_quant_params.scales[0],
               input_quant_params.zero_points[0],
               input_data);
  } else {
    FAKE_DDK_CHECK(input_precision == PrecisionType::FLOAT32)
        << "Unsupported input precision type "
        << static_cast<int>(input_precision) << "!";
  }
  // Dequantize filter data if needs
  auto filter_precision = filter_tensor->attr.precision;
  auto filter_count = shape_production(filter_shape);
  auto& filter_quant_params = filter_tensor->attr.quant_params;
  auto filter_buffer = get_tensor_buffer_address(filter_tensor);
  auto filter_data = reinterpret_cast<float*>(filter_buffer);
  std::vector<float> dequantized_filter_data;
  if (filter_precision == PrecisionType::QUANT_INT8_SYMM_PER_LAYER) {
    dequantized_filter_data.resize(filter_count);
    filter_data = dequantized_filter_data.data();
    dequantize(reinterpret_cast<int8_t*>(filter_buffer),
               filter_shape,
               filter_quant_params.scales[0],
               filter_data);
  } else if (filter_precision == PrecisionType::QUANT_INT8_SYMM_PER_CHANNEL) {
    dequantized_filter_data.resize(filter_count);
    filter_data = dequantized_filter_data.data();
    dequantize(reinterpret_cast<int8_t*>(filter_buffer),
               filter_shape,
               filter_quant_params.scales,
               filter_quant_params.channel_dim,
               filter_data);
  } else if (filter_precision == PrecisionType::QUANT_UINT8_ASYMM_PER_LAYER) {
    dequantized_filter_data.resize(filter_count);
    filter_data = dequantized_filter_data.data();
    dequantize(reinterpret_cast<uint8_t*>(filter_buffer),
               filter_shape,
               filter_quant_params.scales[0],
               filter_quant_params.zero_points[0],
               filter_data);
  } else {
    FAKE_DDK_CHECK(filter_precision == PrecisionType::FLOAT32)
        << "Unsupported filter precision type "
        << static_cast<int>(filter_precision) << "!";
  }
  // Dequantize bias data if needs
  float* bias_data = nullptr;
  std::vector<float> dequantized_bias_data;
  if (bias_tensor) {
    auto& bias_shape = bias_tensor->attr.shape;
    auto bias_count = shape_production(bias_shape);
    auto bias_precision = bias_tensor->attr.precision;
    auto& bias_quant_params = bias_tensor->attr.quant_params;
    FAKE_DDK_CHECK_EQ(bias_count, output_channel_size);
    auto bias_buffer = get_tensor_buffer_address(bias_tensor);
    bias_data = reinterpret_cast<float*>(bias_buffer);
    if (bias_precision == PrecisionType::QUANT_INT32_SYMM_PER_LAYER) {
      dequantized_bias_data.resize(bias_count);
      bias_data = dequantized_bias_data.data();
      dequantize(reinterpret_cast<int32_t*>(bias_buffer),
                 bias_shape,
                 bias_quant_params.scales[0],
                 bias_data);
    } else if (bias_precision == PrecisionType::QUANT_INT32_SYMM_PER_CHANNEL) {
      dequantized_bias_data.resize(bias_count);
      bias_data = dequantized_bias_data.data();
      dequantize(reinterpret_cast<int32_t*>(bias_buffer),
                 bias_shape,
                 bias_quant_params.scales,
                 bias_quant_params.channel_dim,
                 bias_data);
    } else {
      FAKE_DDK_CHECK(bias_precision == PrecisionType::FLOAT32)
          << "Unsupported bias precision type "
          << static_cast<int>(bias_precision) << "!";
    }
  }
  // Update the output shape and allocate/reallocate buffer for the output
  // tensor
  std::vector<int32_t> output_shape = {
      batch_size, output_channel_size, output_height, output_width};
  output_tensor->attr.shape = output_shape;
  auto output_precision = output_tensor->attr.precision;
  auto output_quant_params = output_tensor->attr.quant_params;
  auto output_buffer = get_tensor_buffer_address(output_tensor);
  auto output_count = shape_production(output_shape);
  auto output_data = reinterpret_cast<float*>(output_buffer);
  std::vector<float> dequantized_output_data;
  if (output_precision == PrecisionType::QUANT_INT8_SYMM_PER_LAYER ||
      output_precision == PrecisionType::QUANT_UINT8_ASYMM_PER_LAYER) {
    dequantized_output_data.resize(output_count);
    output_data = dequantized_output_data.data();
  } else {
    FAKE_DDK_CHECK(output_precision == PrecisionType::FLOAT32)
        << "Unsupported output precision type "
        << static_cast<int>(output_precision) << "!";
  }
  for (int bs = 0; bs < batch_size; bs++) {
    for (int g = 0; g < group; g++) {
      for (int oc = 0; oc < output_channel_group; oc++) {
        for (int oh = 0; oh < output_height; oh++) {
          for (int ow = 0; ow < output_width; ow++) {
            int output_index =
                bs * group * output_channel_group * output_height *
                    output_width +
                g * output_channel_group * output_height * output_width +
                oc * output_height * output_width + oh * output_width + ow;
            float output_value =
                bias_data ? bias_data[g * output_channel_group + oc] : 0;
            for (int ic = 0; ic < input_channel_group; ic++) {
              for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                  int iw =
                      ow * stride_width - pad_width_left + kw * dilation_width;
                  int ih = oh * stride_height - pad_height_top +
                           kh * dilation_height;
                  if (iw < 0 || iw >= input_width) continue;
                  if (ih < 0 || ih >= input_height) continue;
                  int input_index =
                      bs * input_channel_size * input_height * input_width +
                      g * input_channel_group * input_height * input_width +
                      ic * input_height * input_width + ih * input_width + iw;
                  int filter_index =
                      g * output_channel_group * input_channel_group *
                          kernel_height * kernel_width +
                      oc * input_channel_group * kernel_height * kernel_width +
                      ic * kernel_height * kernel_width + kh * kernel_width +
                      kw;
                  output_value +=
                      input_data[input_index] * filter_data[filter_index];
                }
              }
            }
            if (fuse_type == FUSE_RELU) {
              output_value = output_value > 0 ? output_value : 0;
            } else if (fuse_type == FUSE_RELU1) {
              output_value = std::min(std::max(0.0f, output_value), 1.0f);
            } else if (fuse_type == FUSE_RELU6) {
              output_value = std::min(std::max(0.0f, output_value), 6.0f);
            } else if (fuse_type == FUSE_NONE) {
            } else {
              return StatusType::FAILURE;
            }
            output_data[output_index] = output_value;
          }
        }
      }
    }
  }
  if (output_precision == PrecisionType::QUANT_INT8_SYMM_PER_LAYER) {
    quantize(output_data,
             output_shape,
             output_quant_params.scales[0],
             reinterpret_cast<int8_t*>(output_buffer));
  } else if (output_precision == PrecisionType::QUANT_UINT8_ASYMM_PER_LAYER) {
    quantize(output_data,
             output_shape,
             output_quant_params.scales[0],
             output_quant_params.zero_points[0],
             reinterpret_cast<uint8_t*>(output_buffer));
  }
  return StatusType::SUCCESS;
}

}  // namespace fake_ddk
