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

#pragma once
#include "../test_include.h"
#include "operators/relu_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
class TestReluOp {
 public:
  explicit TestReluOp(const Program<Dtype> p) : program_(p) {
    if (use_optimize_) {
      to_predict_program_ = program_.optimizeProgram;
    } else {
      to_predict_program_ = program_.originProgram;
    }

    const std::vector<std::shared_ptr<BlockDesc>> blocks =
        to_predict_program_->Blocks();
    //  DLOG << " **block size " << blocks.size();
    for (auto block_desc : blocks) {
      std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
      //    DLOG << " ops " << ops.size();
      for (auto op : ops) {
        if (op->Type() == "relu" &&
            op->Input("X")[0] == "batch_norm_34.tmp_2") {
          DLOG << "in";
          std::shared_ptr<operators::ReluOp<Dtype, float>> test_op =
              std::make_shared<operators::ReluOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope);
          ops_of_block_[*block_desc.get()].push_back(test_op);
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict(const Tensor &t1) {
    // feed
    auto scope = program_.scope;
    Variable *x1_feed_value = scope->Var("batch_norm_34.tmp_2");
    auto tensor_x1 = x1_feed_value->GetMutable<Tensor>();
    tensor_x1->ShareDataWith(t1);

    Variable *output = scope->Var("batch_norm_34.tmp_3");
    auto *output_tensor = output->GetMutable<Tensor>();
    output_tensor->mutable_data<float>({1, 2, 3, 4});

    //  DLOG << typeid(output_tensor).name();
    //  DLOG << "output_tensor dims: " << output_tensor->dims();

    std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
    out_tensor.reset(output_tensor);

    predict(t1, 0);

    return out_tensor;
    // return outvars_tensor;
  }

 private:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;

  void predict(const Tensor &t1, int block_id) {
    std::shared_ptr<BlockDesc> to_predict_block =
        to_predict_program_->Block(block_id);
    for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
      auto op = ops_of_block_[*to_predict_block.get()][j];
      DLOG << "op -> run()";
      op->Run();
    }
  }
};

template class TestReluOp<CPU>;
}  // namespace framework
}  // namespace paddle_mobile

int main() {
  DLOG << "----------**********----------";
  DLOG << "begin to run Relu Test";
  paddle_mobile::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string("../../test/models/mobilenet+ssd"));

  /// input x (1,3,300,300)
  paddle_mobile::framework::Tensor inputx1;
  SetupTensor<float>(&inputx1, {1, 2, 3, 4}, static_cast<float>(-1),
                     static_cast<float>(1));
  auto *inputx1_ptr = inputx1.data<float>();

  paddle_mobile::framework::TestReluOp<paddle_mobile::CPU> testReluOp(program);

  auto output = testReluOp.predict(inputx1);
  auto *output_ptr = output->data<float>();

  for (int i = 0; i < output->numel(); i++) {
    DLOG << output_ptr[i];
  }
  return 0;
}
