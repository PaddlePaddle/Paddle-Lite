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

#include "framework/executor_for_test.h"

template <typename DeviceType, typename OpType>
Executor4Test<DeviceType, OpType>::Executor4Test(const Program<DeviceType> p,
                                                 std::string op_type)
    : Executor<DeviceType>(p) {
  if (this->program_.originProgram == nullptr) {
    LOG(paddle_mobile::LogLevel::kLOG_ERROR)
        << "to_predict_program_ == nullptr";
  }

  const std::vector<std::shared_ptr<BlockDesc>> blocks =
      this->to_predict_program_->Blocks();

  for (std::shared_ptr<BlockDesc> block_desc : blocks) {
    std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
    for (std::shared_ptr<OpDesc> op : ops) {
      if (op->Type() == op_type) {
        std::shared_ptr<OpType> op_ptr = std::make_shared<OpType>(
            op->Type(), op->GetInputs(), op->GetOutputs(), op->GetAttrMap(),
            this->program_.scope);

        this->ops_of_block_[*block_desc.get()].push_back(op_ptr);
        break;
      }
    }
  }
}

template <typename DeviceType, typename OpType>
std::shared_ptr<Tensor> Executor4Test<DeviceType, OpType>::predict(
    const Tensor &t, std::string input, std::string output, const DDim &dDim) {
  auto scope = this->program_.scope;
  Variable *g_feed_value = scope->Var(input);
  auto tensor = g_feed_value->GetMutable<Tensor>();
  tensor->ShareDataWith(t);

  Variable *con_output = scope->Var(output);
  auto *output_tensor = con_output->GetMutable<Tensor>();
  output_tensor->mutable_data<float>(dDim);
  std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
  out_tensor.reset(output_tensor);

  Executor<DeviceType>::predict(t, 0);
  return out_tensor;
}

template class Executor4Test<
    paddle_mobile::CPU,
    paddle_mobile::operators::ConvOp<paddle_mobile::CPU, float>>;
template class Executor4Test<
    paddle_mobile::CPU,
    paddle_mobile::operators::PoolOp<paddle_mobile::CPU, float>>;
template class Executor4Test<
    paddle_mobile::CPU,
    paddle_mobile::operators::SoftmaxOp<paddle_mobile::CPU, float>>;
