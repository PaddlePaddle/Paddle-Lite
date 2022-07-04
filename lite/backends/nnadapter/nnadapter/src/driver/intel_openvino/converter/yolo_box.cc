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
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "operation/yolo_box.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"
namespace nnadapter {
namespace intel_openvino {

int ConvertYoloBox(Converter* converter, core::Operation* operation) {
  YOLO_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto imgsize_tensor = converter->GetMappedTensor(imgsize_operand);
  if (!imgsize_tensor) {
    imgsize_tensor = converter->ConvertOperand(imgsize_operand);
  }

  // Get shape of X.
  auto input_shape = std::make_shared<default_opset::ShapeOf>(
      *input_tensor, GetElementType<int64_t>());
  auto indices_batchsize = default_opset::Constant::create<int32_t>(
      GetElementType<int64_t>(), {1}, {0});
  auto indices_height = default_opset::Constant::create<int32_t>(
      GetElementType<int64_t>(), {1}, {2});
  auto indices_width = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {3});
  auto const_axis0 = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {0});
  // H
  auto input_height = std::make_shared<default_opset::Gather>(
      input_shape, indices_height, const_axis0);
  // W
  auto input_width = std::make_shared<default_opset::Gather>(
      input_shape, indices_width, const_axis0);
  // N
  auto batch_size = std::make_shared<default_opset::Gather>(
      input_shape, indices_batchsize, const_axis0);
  auto const_class_num = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {class_num});
  // Paddle anchors attribute is of type int32. Convert to float for computing
  // convinient.
  std::vector<float> anchors_f32(anchors.begin(), anchors.end());
  uint32_t numanchors = anchors.size() / 2;
  auto const_numanchors = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {numanchors});
  auto default_scale = 1.0f;
  auto const_downsample_ratio = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {downsample_ratio});
  auto scaled_input_height = std::make_shared<default_opset::Multiply>(
      input_height, const_downsample_ratio);
  auto scaled_input_width = std::make_shared<default_opset::Multiply>(
      input_width, const_downsample_ratio);
  // score_shape is {batch_size, input_height * input_width * numanchors,
  // class_num}.
  auto op_mul_whc =
      std::make_shared<default_opset::Multiply>(input_height, input_width);
  op_mul_whc =
      std::make_shared<default_opset::Multiply>(op_mul_whc, const_numanchors);
  auto score_shape = std::make_shared<default_opset::Concat>(
      TensorVector{batch_size, op_mul_whc, const_class_num}, 0);
  auto const_conf_thresh = default_opset::Constant::create<float>(
      GetElementType<float>(), {1}, {conf_thresh});
  // op_x_shape is {batch_size, numanchors, 5 + class_num, input_height,
  // input_width}.
  auto const_class_num_plus5 = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {5 + class_num});
  auto op_x_shape = std::make_shared<default_opset::Concat>(
      TensorVector{batch_size,
                   const_numanchors,
                   const_class_num_plus5,
                   input_height,
                   input_width},
      0);
  auto op_x_reshape = std::make_shared<default_opset::Reshape>(
      *input_tensor, op_x_shape, false);
  auto op_input_order = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {5}, {0, 1, 3, 4, 2});
  auto op_x_transpose =
      std::make_shared<default_opset::Transpose>(op_x_reshape, op_input_order);
  // range x/y
  // range_x: shape {1, input_width} containing 0...input_width
  // range_y: shape {input_height, 1} containing 0...input_height
  auto const_start = default_opset::Constant::create<float>(
      GetElementType<float>(), {}, {0.f});
  auto const_step = default_opset::Constant::create<float>(
      GetElementType<float>(), {}, {1.f});
  auto reduction_axes = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {0});
  auto scaler_input_width = std::make_shared<default_opset::ReduceMin>(
      input_width, reduction_axes, false);
  auto range_x = std::make_shared<default_opset::Range>(
      const_start, scaler_input_width, const_step, GetElementType<float>());
  auto op_range_x = std::make_shared<default_opset::Unsqueeze>(
      range_x,
      default_opset::Constant::create<int64_t>(
          GetElementType<int64_t>(), {1}, {0}));
  auto scaler_input_height = std::make_shared<default_opset::ReduceMin>(
      input_height, reduction_axes, false);
  auto range_y = std::make_shared<default_opset::Range>(
      const_start, scaler_input_height, const_step, GetElementType<float>());
  auto op_range_y = std::make_shared<default_opset::Unsqueeze>(
      range_y,
      default_opset::Constant::create<int64_t>(
          GetElementType<int64_t>(), {1}, {1}));
  auto op_range_x_shape = std::make_shared<default_opset::Concat>(
      TensorVector{default_opset::Constant::create<int64_t>(
                       GetElementType<int64_t>(), {1}, {1}),
                   input_width},
      0);
  auto op_range_y_shape = std::make_shared<default_opset::Concat>(
      TensorVector{input_height,
                   default_opset::Constant::create<int64_t>(
                       GetElementType<int64_t>(), {1}, {1})},
      0);
  // Shape (H, W)
  auto op_grid_x =
      std::make_shared<default_opset::Tile>(op_range_x, op_range_y_shape);
  auto op_grid_y =
      std::make_shared<default_opset::Tile>(op_range_y, op_range_x_shape);

  auto op_split_axis = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {-1});
  auto op_split_lengths = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {6}, {1, 1, 1, 1, 1, class_num});
  auto op_split_input = std::make_shared<default_opset::VariadicSplit>(
      op_x_transpose, op_split_axis, op_split_lengths);
  // shape (batch_size, numanchors, H, W, 1)
  auto op_box_x = op_split_input->output(0);
  auto op_box_y = op_split_input->output(1);
  auto op_box_w = op_split_input->output(2);
  auto op_box_h = op_split_input->output(3);
  auto op_conf = op_split_input->output(4);
  auto op_prob = op_split_input->output(5);
  // x/y
  std::shared_ptr<Operator> op_box_x_sigmoid =
      std::make_shared<default_opset::Sigmoid>(op_box_x);
  // Float not-equal
  std::shared_ptr<Operator> op_box_y_sigmoid =
      std::make_shared<default_opset::Sigmoid>(op_box_y);
  if (std::fabs(scale_x_y - default_scale) > 1e-6) {
    float bias_x_y = -0.5 * (scale_x_y - 1.0);
    auto scale_x_y_op = default_opset::Constant::create<float>(
        GetElementType<float>(), {1}, {scale_x_y});
    auto bias_x_y_op = default_opset::Constant::create<float>(
        GetElementType<float>(), {1}, {bias_x_y});
    op_box_x_sigmoid = std::make_shared<default_opset::Multiply>(
        op_box_x_sigmoid, scale_x_y_op);
    op_box_x_sigmoid =
        std::make_shared<default_opset::Add>(op_box_x_sigmoid, bias_x_y_op);
    op_box_y_sigmoid = std::make_shared<default_opset::Multiply>(
        op_box_y_sigmoid, scale_x_y_op);
    op_box_y_sigmoid =
        std::make_shared<default_opset::Add>(op_box_y_sigmoid, bias_x_y_op);
  }
  auto squeeze_box_x = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {4});
  auto op_box_x_squeeze =
      std::make_shared<default_opset::Squeeze>(op_box_x_sigmoid, squeeze_box_x);
  auto squeeze_box_y = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {4});
  auto op_box_y_squeeze =
      std::make_shared<default_opset::Squeeze>(op_box_y_sigmoid, squeeze_box_y);
  auto op_box_x_add_grid =
      std::make_shared<default_opset::Add>(op_grid_x, op_box_x_squeeze);
  auto op_box_y_add_grid =
      std::make_shared<default_opset::Add>(op_grid_y, op_box_y_squeeze);
  auto op_input_h = std::make_shared<default_opset::Convert>(
      input_height, GetElementType<float>());
  auto op_input_w = std::make_shared<default_opset::Convert>(
      input_width, GetElementType<float>());
  auto op_box_x_encode =
      std::make_shared<default_opset::Divide>(op_box_x_add_grid, op_input_w);
  auto op_box_y_encode =
      std::make_shared<default_opset::Divide>(op_box_y_add_grid, op_input_h);
  // w/h
  auto op_anchor_tensor = default_opset::Constant::create<float>(
      GetElementType<float>(), {numanchors, 2}, anchors_f32);
  auto split_axis = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {}, {1});
  auto op_anchor_split =
      std::make_shared<default_opset::Split>(op_anchor_tensor, split_axis, 2);
  auto op_anchor_w_origin = op_anchor_split->output(0);
  auto op_anchor_h_origin = op_anchor_split->output(1);
  auto float_input_height = std::make_shared<default_opset::Convert>(
      scaled_input_height, GetElementType<float>());
  auto op_anchor_h = std::make_shared<default_opset::Divide>(
      op_anchor_h_origin, float_input_height);
  auto float_input_width = std::make_shared<default_opset::Convert>(
      scaled_input_width, GetElementType<float>());
  auto op_anchor_w = std::make_shared<default_opset::Divide>(op_anchor_w_origin,
                                                             float_input_width);
  auto op_new_anchor_shape = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {4}, {1, numanchors, 1, 1});
  auto op_anchor_w_reshape = std::make_shared<default_opset::Reshape>(
      op_anchor_w, op_new_anchor_shape, false);
  auto op_anchor_h_reshape = std::make_shared<default_opset::Reshape>(
      op_anchor_h, op_new_anchor_shape, false);
  auto squeeze_box_wh = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {4});
  auto op_box_w_squeeze =
      std::make_shared<default_opset::Squeeze>(op_box_w, squeeze_box_wh);
  auto op_box_h_squeeze =
      std::make_shared<default_opset::Squeeze>(op_box_h, squeeze_box_wh);
  auto op_box_w_exp = std::make_shared<default_opset::Exp>(op_box_w_squeeze);
  auto op_box_h_exp = std::make_shared<default_opset::Exp>(op_box_h_squeeze);
  auto op_box_w_encode = std::make_shared<default_opset::Multiply>(
      op_box_w_exp, op_anchor_w_reshape);
  auto op_box_h_encode = std::make_shared<default_opset::Multiply>(
      op_box_h_exp, op_anchor_h_reshape);
  // Confidence
  auto op_conf_sigmoid = std::make_shared<default_opset::Sigmoid>(op_conf);
  auto op_concat = std::make_shared<default_opset::Concat>(
      TensorVector{default_opset::Constant::create<int64_t>(
                       GetElementType<int64_t>(), {1}, {1}),
                   const_numanchors,
                   input_height,
                   input_width,
                   default_opset::Constant::create<int64_t>(
                       GetElementType<int64_t>(), {1}, {1})},
      0);
  // {1, numanchors, input_height, input_width, 1}
  auto op_conf_thresh =
      std::make_shared<default_opset::Broadcast>(const_conf_thresh, op_concat);
  auto op_conf_sub = std::make_shared<default_opset::Subtract>(op_conf_sigmoid,
                                                               op_conf_thresh);
  auto op_conf_clip = std::make_shared<default_opset::Clamp>(
      op_conf_sub, 0.0f, std::numeric_limits<float>::max());
  auto op_zeros =
      default_opset::Constant::create<float>(GetElementType<float>(), {1}, {0});
  auto op_conf_clip_bool =
      std::make_shared<default_opset::Greater>(op_conf_clip, op_zeros);
  auto op_conf_clip_cast = std::make_shared<default_opset::Convert>(
      op_conf_clip_bool, GetElementType<float>());
  auto op_conf_set_zero = std::make_shared<default_opset::Multiply>(
      op_conf_sigmoid, op_conf_clip_cast);
  // Probability.
  auto op_prob_sigmoid = std::make_shared<default_opset::Sigmoid>(op_prob);
  auto op_new_shape = std::make_shared<default_opset::Concat>(
      TensorVector{batch_size,
                   const_numanchors,
                   input_height,
                   input_width,
                   default_opset::Constant::create<int64_t>(
                       GetElementType<int64_t>(), {1}, {1})},
      0);
  // {batch_size, int(numanchors), input_height, input_width, 1}
  auto op_conf_new_shape = std::make_shared<default_opset::Reshape>(
      op_conf_set_zero, op_new_shape, false);
  // Broadcast confidence * probability of each category
  auto op_score = std::make_shared<default_opset::Multiply>(op_prob_sigmoid,
                                                            op_conf_new_shape);
  // For bbox which has object (greater than threshold)
  auto op_conf_bool =
      std::make_shared<default_opset::Greater>(op_conf_new_shape, op_zeros);
  auto op_box_x_new_shape = std::make_shared<default_opset::Reshape>(
      op_box_x_encode, op_new_shape, false);
  auto op_box_y_new_shape = std::make_shared<default_opset::Reshape>(
      op_box_y_encode, op_new_shape, false);
  auto op_box_w_new_shape = std::make_shared<default_opset::Reshape>(
      op_box_w_encode, op_new_shape, false);
  auto op_box_h_new_shape = std::make_shared<default_opset::Reshape>(
      op_box_h_encode, op_new_shape, false);
  auto op_pred_box =
      std::make_shared<default_opset::Concat>(TensorVector{op_box_x_new_shape,
                                                           op_box_y_new_shape,
                                                           op_box_w_new_shape,
                                                           op_box_h_new_shape},
                                              4);
  auto op_conf_cast = std::make_shared<default_opset::Convert>(
      op_conf_bool, GetElementType<float>());
  auto op_pred_box_mul_conf =
      std::make_shared<default_opset::Multiply>(op_pred_box, op_conf_cast);
  auto op_box_shape = std::make_shared<default_opset::Concat>(
      TensorVector{batch_size,
                   op_mul_whc,
                   default_opset::Constant::create<int64_t>(
                       GetElementType<int64_t>(), {1}, {4})},
      0);
  // {batch_size, int(numanchors) * input_height * input_width, 4}
  auto op_pred_box_new_shape = std::make_shared<default_opset::Reshape>(
      op_pred_box_mul_conf, op_box_shape, false);
  auto pred_box_split_axis = default_opset::Constant::create<int32_t>(
      GetElementType<int64_t>(), {}, {2});
  auto op_pred_box_split = std::make_shared<default_opset::Split>(
      op_pred_box_new_shape, pred_box_split_axis, 4);
  auto op_pred_box_x = op_pred_box_split->output(0);
  auto op_pred_box_y = op_pred_box_split->output(1);
  auto op_pred_box_w = op_pred_box_split->output(2);
  auto op_pred_box_h = op_pred_box_split->output(3);
  // x,y,w,h -> x1,y1,x2,y2
  auto op_number_two = default_opset::Constant::create<float>(
      GetElementType<float>(), {1}, {2.0f});
  auto op_half_w =
      std::make_shared<default_opset::Divide>(op_pred_box_w, op_number_two);
  auto op_half_h =
      std::make_shared<default_opset::Divide>(op_pred_box_h, op_number_two);
  auto op_pred_box_x1 =
      std::make_shared<default_opset::Subtract>(op_pred_box_x, op_half_w);
  auto op_pred_box_y1 =
      std::make_shared<default_opset::Subtract>(op_pred_box_y, op_half_h);
  auto op_pred_box_x2 =
      std::make_shared<default_opset::Add>(op_pred_box_x, op_half_w);
  auto op_pred_box_y2 =
      std::make_shared<default_opset::Add>(op_pred_box_y, op_half_h);
  // Map normalized coords to original image.
  auto indices_height_imgsize = default_opset::Constant::create<int32_t>(
      GetElementType<int64_t>(), {1}, {0});
  auto indices_width_imgsize = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {1});
  auto const_axis1 = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {1});
  // shape_*imgsize_tenosr[0]
  auto op_img_height = std::make_shared<default_opset::Gather>(
      *imgsize_tensor, indices_height_imgsize, const_axis1);
  // shape_ * imgsize_tenosr[1]
  auto op_img_width = std::make_shared<default_opset::Gather>(
      *imgsize_tensor, indices_width_imgsize, const_axis1);
  auto op_img_width_cast = std::make_shared<default_opset::Convert>(
      op_img_width, GetElementType<float>());
  auto op_img_height_cast = std::make_shared<default_opset::Convert>(
      op_img_height, GetElementType<float>());
  auto squeeze_axes2 = default_opset::Constant::create<int64_t>(
      GetElementType<int64_t>(), {1}, {2});
  // Reshape (N,C,1) -> (N,C) for upcomping multiply.
  auto op_pred_box_x1_reshape =
      std::make_shared<default_opset::Squeeze>(op_pred_box_x1, squeeze_axes2);
  auto op_pred_box_y1_reshape =
      std::make_shared<default_opset::Squeeze>(op_pred_box_y1, squeeze_axes2);
  auto op_pred_box_x2_reshape =
      std::make_shared<default_opset::Squeeze>(op_pred_box_x2, squeeze_axes2);
  auto op_pred_box_y2_reshape =
      std::make_shared<default_opset::Squeeze>(op_pred_box_y2, squeeze_axes2);
  auto op_pred_box_x1_squeeze = std::make_shared<default_opset::Multiply>(
      op_pred_box_x1_reshape, op_img_width_cast);
  auto op_pred_box_y1_squeeze = std::make_shared<default_opset::Multiply>(
      op_pred_box_y1_reshape, op_img_height_cast);
  auto op_pred_box_x2_squeeze = std::make_shared<default_opset::Multiply>(
      op_pred_box_x2_reshape, op_img_width_cast);
  auto op_pred_box_y2_squeeze = std::make_shared<default_opset::Multiply>(
      op_pred_box_y2_reshape, op_img_height_cast);
  std::shared_ptr<Operator> op_pred_box_result;
  if (clip_bbox) {
    auto op_number_one = default_opset::Constant::create<float>(
        GetElementType<float>(), {1}, {1.0});
    auto op_new_img_height = std::make_shared<default_opset::Subtract>(
        op_img_height_cast, op_number_one);
    auto op_new_img_width = std::make_shared<default_opset::Subtract>(
        op_img_width_cast, op_number_one);
    // x2 - (w-1)
    auto op_pred_box_x2_sub_w = std::make_shared<default_opset::Subtract>(
        op_pred_box_x2_squeeze, op_new_img_width);
    // y2 - (h-1)
    auto op_pred_box_y2_sub_h = std::make_shared<default_opset::Subtract>(
        op_pred_box_y2_squeeze, op_new_img_height);
    auto max_const = std::numeric_limits<float>::max();
    auto op_pred_box_x1_clip = std::make_shared<default_opset::Clamp>(
        op_pred_box_x1_squeeze, 0.0f, max_const);
    auto op_pred_box_y1_clip = std::make_shared<default_opset::Clamp>(
        op_pred_box_y1_squeeze, 0.0f, max_const);
    auto op_pred_box_x2_clip = std::make_shared<default_opset::Clamp>(
        op_pred_box_x2_sub_w, 0.0f, max_const);
    auto op_pred_box_y2_clip = std::make_shared<default_opset::Clamp>(
        op_pred_box_y2_sub_h, 0.0f, max_const);
    auto op_pred_box_x2_res = std::make_shared<default_opset::Subtract>(
        op_pred_box_x2_squeeze, op_pred_box_x2_clip);
    auto op_pred_box_y2_res = std::make_shared<default_opset::Subtract>(
        op_pred_box_y2_squeeze, op_pred_box_y2_clip);
    // Reshape back to (N,C,1).
    auto op_pred_box_x1_clip2 = std::make_shared<default_opset::Unsqueeze>(
        op_pred_box_x1_clip, squeeze_axes2);
    auto op_pred_box_y1_clip2 = std::make_shared<default_opset::Unsqueeze>(
        op_pred_box_y1_clip, squeeze_axes2);
    auto op_pred_box_x2_res2 = std::make_shared<default_opset::Unsqueeze>(
        op_pred_box_x2_res, squeeze_axes2);
    auto op_pred_box_y2_res2 = std::make_shared<default_opset::Unsqueeze>(
        op_pred_box_y2_res, squeeze_axes2);
    op_pred_box_result = std::make_shared<default_opset::Concat>(
        TensorVector{op_pred_box_x1_clip2,
                     op_pred_box_y1_clip2,
                     op_pred_box_x2_res2,
                     op_pred_box_y2_res2},
        -1);
  } else {
    // Reshape back to (N,C,1).
    auto op_pred_box_x1_decode = std::make_shared<default_opset::Unsqueeze>(
        op_pred_box_x1_squeeze, squeeze_axes2);
    auto op_pred_box_y1_decode = std::make_shared<default_opset::Unsqueeze>(
        op_pred_box_y1_squeeze, squeeze_axes2);
    auto op_pred_box_x2_decode = std::make_shared<default_opset::Unsqueeze>(
        op_pred_box_x2_squeeze, squeeze_axes2);
    auto op_pred_box_y2_decode = std::make_shared<default_opset::Unsqueeze>(
        op_pred_box_y2_squeeze, squeeze_axes2);
    op_pred_box_result = std::make_shared<default_opset::Concat>(
        TensorVector{op_pred_box_x1_decode,
                     op_pred_box_y1_decode,
                     op_pred_box_x2_decode,
                     op_pred_box_y2_decode},
        -1);
  }
  auto op_score_new_shape =
      std::make_shared<default_opset::Reshape>(op_score, score_shape, false);
  MAP_OUTPUT(boxes_operand, op_pred_box_result, 0);
  MAP_OUTPUT(scores_operand, op_score_new_shape, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
