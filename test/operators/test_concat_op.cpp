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
#include "operators/concat_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype> class TestConcatOp {
public:
  explicit TestConcatOp(const Program<Dtype> p) : program_(p) {
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
        if (op->Type() == "concat" && op->Input("X")[0] == "conv2d_3.tmp_1") {
          DLOG << " mul attr size: " << op->GetAttrMap().size();
          DLOG << " inputs size: " << op->GetInputs().size();
          DLOG << " outputs size: " << op->GetOutputs().size();
          DLOG << " Input X is : " << op->Input("X")[0];
          DLOG << " Output Out is : " << op->Output("Out")[0];
          DLOG << " axis : " << op->GetAttrMap().at("axis").Get<int>();

          std::shared_ptr<operators::ConcatOp<Dtype, float>> concat =
              std::make_shared<operators::ConcatOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope);
          ops_of_block_[*block_desc.get()].push_back(concat);
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict_concat(Tensor &t1, Tensor &t2, Tensor &t3,
                                         Tensor &t4) {
    // feed
    auto scope = program_.scope;
    Variable *x1_feed_value = scope->Var("conv2d_3.tmp_1");
    auto tensor_x1 = x1_feed_value->GetMutable<Tensor>();
    tensor_x1->ShareDataWith(t1);

    Variable *x2_feed_value = scope->Var("conv2d_5.tmp_1");
    auto tensor_x2 = x2_feed_value->GetMutable<Tensor>();
    tensor_x2->ShareDataWith(t2);

    Variable *x3_feed_value = scope->Var("conv2d_7.tmp_1");
    auto tensor_x3 = x3_feed_value->GetMutable<Tensor>();
    tensor_x3->ShareDataWith(t3);

    Variable *x4_feed_value = scope->Var("conv2d_8.tmp_1");
    auto tensor_x4 = x4_feed_value->GetMutable<Tensor>();
    tensor_x4->ShareDataWith(t4);

    Variable *con_output = scope->Var("concat_0.tmp_0");
    auto *output_tensor = con_output->GetMutable<Tensor>();
    output_tensor->mutable_data<float>({4, 100, 2, 2});
    //  DLOG << typeid(output_tensor).name();
    //  DLOG << "output_tensor dims: " << output_tensor->dims();

    std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
    out_tensor.reset(output_tensor);

    predict_concat(t1, t2, t3, t4, 0);
    return out_tensor;
  }

private:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;

  void predict_concat(const Tensor &t1, const Tensor &t2, const Tensor &t3,
                      const Tensor &t4, int block_id) {
    std::shared_ptr<BlockDesc> to_predict_block =
        to_predict_program_->Block(block_id);
    for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
      auto op = ops_of_block_[*to_predict_block.get()][j];
      DLOG << "op -> run()";
      op->Run();
    }
  }
};

template class TestConcatOp<CPU>;
} // namespace framework
} // namespace paddle_mobile

int main() {
  DLOG << "----------**********----------";
  DLOG << "begin to run ConcatOp Test";
  paddle_mobile::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string("../../test/models/googlenet"));

  /// input x (4,10,2,2)
  paddle_mobile::framework::Tensor inputx1;
  SetupTensor<float>(&inputx1, {4, 10, 2, 2}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx1_ptr = inputx1.data<float>();
  /// input x (4,20,2,2)
  paddle_mobile::framework::Tensor inputx2;
  SetupTensor<float>(&inputx2, {4, 20, 2, 2}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx2_ptr = inputx2.data<float>();
  /// input x (4,30,2,2)
  paddle_mobile::framework::Tensor inputx3;
  SetupTensor<float>(&inputx3, {4, 30, 2, 2}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx3_ptr = inputx3.data<float>();
  /// input x (4,40,2,2)
  paddle_mobile::framework::Tensor inputx4;
  SetupTensor<float>(&inputx4, {4, 40, 2, 2}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx4_ptr = inputx4.data<float>();

  paddle_mobile::framework::TestConcatOp<paddle_mobile::CPU> testConcatOp(
      program);

  auto output_concat =
      testConcatOp.predict_concat(inputx1, inputx2, inputx3, inputx4);
  auto *output_concat_ptr = output_concat->data<float>();

  int input_n = 1;
  int input_c = 2;
  int input_h = 0;
  int input_w = 1;
  int stride0 = inputx3.numel() / inputx3.dims()[0];
  int stride1 = inputx3.numel() / inputx3.dims()[0] / inputx3.dims()[1];
  int stride2 = inputx3.dims()[3];
  /// inputx1 (4,10,2,2),
  /// inputx2 (4,20,2,2),
  /// inputx3 (4,30,2,2),
  /// inputx4 (4,40,2,2),
  /// axis = 1
  /// output (4,100,2,2)
  int input_index =
      input_n * stride0 + input_c * stride1 + input_h * stride2 + input_w;
  int output_index = input_n * 100 * 2 * 2 +
                     (input_c + inputx1.dims()[1] + inputx2.dims()[1]) * 2 * 2 +
                     input_h * 2 + input_w;

  DLOG << " inputx3[1,2,0,1] = " << inputx3_ptr[input_index];
  DLOG << " output[1,12,0,1] = " << output_concat_ptr[output_index];
  return 0;
}
