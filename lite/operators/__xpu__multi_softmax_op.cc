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

#include "lite/operators/__xpu__multi_softmax_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUMultiSoftmaxOp::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  return true;
}

bool XPUMultiSoftmaxOp::InferShapeImpl() const {
  auto in_dims = param_.input->dims();
  auto dim_size = in_dims.size();
  CHECK_EQ(dim_size, 2);
  auto lod = param_.lod;
  const auto &outs = param_.output;
  if (param_.concat_output != nullptr) {
    param_.concat_output->Resize(in_dims);
  }
  const int outs_number = outs.size();
  std::vector<lite::DDim> outs_dims;
  outs_dims.reserve(outs_number);
  for (size_t i = 0; i < lod.size() - 1; i++) {
    auto dim = in_dims;
    dim[1] = lod[i + 1] - lod[i];
    outs_dims.push_back(dim);
  }
  for (size_t j = 0; j < outs_dims.size(); ++j) {
    outs[j]->Resize(outs_dims[j]);
    outs[j]->set_lod(param_.input->lod());
  }
  return true;
}

bool XPUMultiSoftmaxOp::AttachImpl(const cpp::OpDesc &opdesc,
                                   lite::Scope *scope) {
  param_.input = scope->FindTensor(opdesc.Input("Input").front());
  param_.output.clear();
  auto Outputs = opdesc.Output("Output");
  for (auto var : Outputs) {
    param_.output.emplace_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  param_.lod = opdesc.GetAttr<std::vector<int>>("lod");

  std::vector<std::string> output_arg_names = opdesc.OutputArgumentNames();
  if (std::find(output_arg_names.begin(),
                output_arg_names.end(),
                "ConcatOut") != output_arg_names.end()) {
    auto arguments = opdesc.Output("ConcatOut");
    if (arguments.size() > 0) {
      auto arg_var = scope->FindVar(arguments.front());
      if (arg_var != nullptr) {
        param_.concat_output =
            const_cast<lite::Tensor *>(&(arg_var->Get<lite::Tensor>()));
      }
    }
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__multi_softmax,
                 paddle::lite::operators::XPUMultiSoftmaxOp);
