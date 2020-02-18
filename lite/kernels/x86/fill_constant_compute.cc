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

#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class FillConstantCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::FillConstantParam;

  inline DDimLite GetShape(const param_t& param) {
    // 1. shape is a Tensor
    if (param.shape_tensor != nullptr) {
      auto* shape_tensor = param.shape_tensor;
      auto* shape_data = shape_tensor->data<int>();
      auto vec_shape =
          std::vector<int64_t>(shape_data, shape_data + shape_tensor->numel());
      return DDimLite(vec_shape);
    }

    // 2. shape is a list/tuple containing Tensor
    auto shape_tensor_list = param.shape_tensor_list;
    if (shape_tensor_list.size() > 0) {
      std::vector<int64_t> vec_shape;
      for (size_t i = 0; i < shape_tensor_list.size(); ++i) {
        auto tensor = shape_tensor_list[i];
        vec_shape.push_back(*tensor->data<int>());
      }
      return DDimLite(vec_shape);
    }

    // 3. shape is a list/tuple without containing Tensor
    auto vec_shape = param.shape;
    return DDimLite(vec_shape);
  }

  void PrepareForRun() override {
    auto& param = *param_.get_mutable<param_t>();
    auto outdims = GetShape(param);
    param.Out->Resize(outdims);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    CHECK(context.x86_device_context());

    param.Out->template mutable_data<T>();

    paddle::operators::math::set_constant(
        *context.x86_device_context(), &param.Out->raw_tensor(), param.value);
  }

  virtual ~FillConstantCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// float
REGISTER_LITE_KERNEL(fill_constant,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::FillConstantCompute<float>,
                     def)
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("ShapeTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
