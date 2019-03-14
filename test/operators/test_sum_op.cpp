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

#include "../test_helper.h"
#include "../test_include.h"
#include "operators/sum_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
class TestSumOp {
 public:
  explicit TestSumOp(const Program<Dtype> p) : program_(p) {
    if (use_optimize_) {
      to_predict_program_ = program_.optimizeProgram;
    } else {
      to_predict_program_ = program_.originProgram;
    }

    const std::vector<std::shared_ptr<BlockDesc>> blocks =
        to_predict_program_->Blocks();
    //  DLOG << " **block size " << blocks.size();
    for (int i = 0; i < blocks.size(); ++i) {
      std::shared_ptr<BlockDesc> block_desc = blocks[i];
      std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
      //    DLOG << " ops " << ops.size();
      for (int j = 0; j < ops.size(); ++j) {
        std::shared_ptr<OpDesc> op = ops[j];
        if (op->Type() == "sum" && op->Input("X")[0] == "fc_2.tmp_0") {
          DLOG << " sum attr size: " << op->GetAttrMap().size();
          DLOG << " inputs size: " << op->GetInputs().size();
          DLOG << " outputs size: " << op->GetOutputs().size();

          std::shared_ptr<operators::SumOp<Dtype, float>> lrn =
              std::make_shared<operators::SumOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope.get());
          ops_of_block_[*block_desc.get()].push_back(lrn);
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict_bn(const Tensor &t1, const Tensor &t2) {
    // feed
    auto scope = program_.scope.get();
    Variable *x1_feed_value = scope->Var("fc_2.tmp_0");
    auto tensor_x1 = x1_feed_value->GetMutable<LoDTensor>();
    tensor_x1->ShareDataWith(t1);

    Variable *x2_feed_value = scope->Var("fc_2.tmp_1");
    auto tensor_x2 = x2_feed_value->GetMutable<LoDTensor>();
    tensor_x2->ShareDataWith(t2);

    Variable *output = scope->Var("fc_2.tmp_2");
    auto *output_tensor = output->GetMutable<LoDTensor>();
    output_tensor->mutable_data<float>({2, 96});
    //  DLOG << typeid(output_tensor).name();
    //  DLOG << "output_tensor dims: " << output_tensor->dims();

    std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
    out_tensor.reset(output_tensor);

    predict_bn(t1, t2, 0);
    return out_tensor;
  }

 private:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;

  void predict_bn(const Tensor &t1, const Tensor &t2, int block_id) {
    std::shared_ptr<BlockDesc> to_predict_block =
        to_predict_program_->Block(block_id);
    for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
      auto op = ops_of_block_[*to_predict_block.get()][j];
      DLOG << "op -> run()";
      op->Run();
    }
  }
};

template class TestSumOp<CPU>;
}  // namespace framework
}  // namespace paddle_mobile

int main() {
  DLOG << "----------**********----------";
  DLOG << "begin to run Sum Test";
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string(g_eng) + "/model",
                             std::string(g_eng) + "/params");

  /// input x (4,10,2,2)
  paddle_mobile::framework::Tensor inputx1;
  SetupTensor<float>(&inputx1, {2, 96}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx1_ptr = inputx1.data<float>();

  paddle_mobile::framework::Tensor inputx2;
  SetupTensor<float>(&inputx2, {2, 96}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx2_ptr = inputx2.data<float>();

  paddle_mobile::framework::TestSumOp<paddle_mobile::CPU> testSumOp(program);

  auto output_sum = testSumOp.predict_bn(inputx1, inputx2);
  auto *output_sum_ptr = output_sum->data<float>();

  DLOG << "input1 44: " << inputx1_ptr[44];
  DLOG << "input2 44: " << inputx2_ptr[44];
  DLOG << "out 44 :" << output_sum_ptr[44];

  return 0;
}
