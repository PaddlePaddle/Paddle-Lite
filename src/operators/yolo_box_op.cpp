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

#include "operators/yolo_box_op.h"
#include <vector>

namespace paddle_mobile {
namespace operators {

#ifdef YOLOBOX_OP
template <typename Dtype, typename T>
void YoloBoxOp<Dtype, T>::InferShape() const {
  auto dim_x = this->param_.Input()->dims();
  auto dim_imgsize = this->param_.ImgSize()->dims();
  auto anchors = this->param_.Anchors();
  int anchor_num = anchors.size() / 2;
  auto class_num = this->param_.ClassNum();

  // PADDLE_ENFORCE_EQ(dim_x.size(), 4, "Input(X) should be a 4-D tensor.");
  // PADDLE_ENFORCE_EQ(
  //     dim_x[1], anchor_num * (5 + class_num),
  //     "Input(X) dim[1] should be equal to (anchor_mask_number * (5 "
  //     "+ class_num)).");
  // PADDLE_ENFORCE_EQ(dim_imgsize.size(), 2,
  //                   "Input(ImgSize) should be a 2-D tensor.");
  // PADDLE_ENFORCE_EQ(
  //     dim_imgsize[0], dim_x[0],
  //     "Input(ImgSize) dim[0] and Input(X) dim[0] should be same.");
  // PADDLE_ENFORCE_EQ(dim_imgsize[1], 2, "Input(ImgSize) dim[1] should be 2.");
  // PADDLE_ENFORCE_GT(anchors.size(), 0,
  //                   "Attr(anchors) length should be greater than 0.");
  // PADDLE_ENFORCE_EQ(anchors.size() % 2, 0,
  //                   "Attr(anchors) length should be even integer.");
  // PADDLE_ENFORCE_GT(class_num, 0,
  //                   "Attr(class_num) should be an integer greater than 0.");

  int box_num = dim_x[2] * dim_x[3] * anchor_num;
  std::vector<int32_t> dim_boxes({dim_x[0], box_num, 4});
  std::vector<int32_t> dim_scores({dim_x[0], box_num, class_num});

  this->param_.OutputBoxes()->Resize(framework::make_ddim(dim_boxes));
  this->param_.OutputScores()->Resize(framework::make_ddim(dim_scores));
}
#endif  // YOLOBOX_OP

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CPU
#ifdef YOLOBOX_OP
REGISTER_OPERATOR_CPU(yolo_box, ops::YoloBoxOp);
#endif  // YOLOBOX_OP
#endif  // PADDLE_MOBILE_CPU

#ifdef PADDLE_MOBILE_CL
#ifdef YOLOBOX_OP
REGISTER_OPERATOR_CL(yolo_box, ops::YoloBoxOp);
#endif  // YOLOBOX_OP
#endif  // PADDLE_MOBILE_CL

#ifdef YOLOBOX_OP
#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(yolo_box, ops::YoloBoxOp);
#endif
#endif
