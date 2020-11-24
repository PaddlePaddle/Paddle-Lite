// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/range_op.h"
#include <cmath>
#include <functional>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool RangeOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.Start);
  CHECK_OR_FALSE(param_.End);
  CHECK_OR_FALSE(param_.Step);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

template <typename T>
void GetSize(T start, T end, T step, int64_t* size) {
  CHECK(!std::equal_to<T>()(step, 0))
      << "The step of range op should not be 0.";
  CHECK(((start < end) && (step > 0)) || ((start > end) && (step < 0)))
      << "The step should be greater than 0 while start < end. And the "
         "step should be less than 0 while start > end.";
  *size = std::is_integral<T>::value
              ? ((std::abs(end - start) + std::abs(step) - 1) / std::abs(step))
              : std::ceil(std::abs((end - start) / step));
}

bool RangeOpLite::InferShapeImpl() const {
  int64_t size = 0;
  switch (param_.Start->precision()) {
    case PRECISION(kFloat):
      GetSize(param_.Start->data<float>()[0],
              param_.End->data<float>()[0],
              param_.Step->data<float>()[0],
              &size);
      break;
    case PRECISION(kInt32):
      GetSize(param_.Start->data<int>()[0],
              param_.End->data<int>()[0],
              param_.Step->data<int>()[0],
              &size);
      break;
    default:
      LOG(FATAL) << "not supported precision type of range: "
                 << lite_api::PrecisionToStr(param_.Start->precision());
      break;
  }
  param_.Out->Resize(std::vector<int64_t>({size}));
  return true;
}

bool RangeOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto start = opdesc.Input("Start").front();
  auto end = opdesc.Input("End").front();
  auto step = opdesc.Input("Step").front();
  auto out = opdesc.Output("Out").front();

  param_.Start = scope->FindTensor(start);
  param_.End = scope->FindTensor(end);
  param_.Step = scope->FindTensor(step);
  param_.Out = scope->FindMutableTensor(out);

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(range, paddle::lite::operators::RangeOpLite);
