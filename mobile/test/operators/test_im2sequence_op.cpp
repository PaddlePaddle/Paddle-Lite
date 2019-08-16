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
#include "operators/im2sequence_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
class TestIm2SequenceOp {
 public:
  explicit TestIm2SequenceOp(const Program<Dtype> p) : program_(p) {
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
        if (op->Type() == "im2sequence" &&
            op->Input("X")[0] == "conv2d_19.tmp_1") {
          DLOG << " im2squence attr size: " << op->GetAttrMap().size();
          DLOG << " inputs size: " << op->GetInputs().size();
          DLOG << " outputs size: " << op->GetOutputs().size();

          std::shared_ptr<operators::Im2SequenceOp<Dtype, float>> lrn =
              std::make_shared<operators::Im2SequenceOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope.get());
          ops_of_block_[*block_desc.get()].push_back(lrn);
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict_bn(const Tensor &t1) {
    // feed
    auto scope = program_.scope.get();
    Variable *x1_feed_value = scope->Var("conv2d_19.tmp_1");
    auto tensor_x1 = x1_feed_value->GetMutable<LoDTensor>();
    tensor_x1->ShareDataWith(t1);
    Variable *output = scope->Var("im2sequence_0.tmp_0");
    auto *output_tensor = output->GetMutable<LoDTensor>();
    output_tensor->mutable_data<float>({2, 12});
    //  DLOG << typeid(output_tensor).name();
    //  DLOG << "output_tensor dims: " << output_tensor->dims();

    std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
    out_tensor.reset(output_tensor);

    predict_bn(t1, 0);
    return out_tensor;
  }

 private:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;

  void predict_bn(const Tensor &t1, int block_id) {
    std::shared_ptr<BlockDesc> to_predict_block =
        to_predict_program_->Block(block_id);
    for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
      auto op = ops_of_block_[*to_predict_block.get()][j];
      DLOG << "op -> run()";
      op->Run();
    }
  }
};

template class TestIm2SequenceOp<CPU>;
}  // namespace framework
}  // namespace paddle_mobile

int main() {
  DLOG << "----------**********----------";
  DLOG << "begin to run Im2Sequence Test";
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string(g_eng) + "/model",
                             std::string(g_eng) + "/params");

  /// input x (4,10,2,2)
  paddle_mobile::framework::Tensor inputx;
  SetupTensor<float>(&inputx, {1, 2, 6, 2}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx_ptr = inputx.data<float>();

  paddle_mobile::framework::TestIm2SequenceOp<paddle_mobile::CPU>
      testIm2SequenceOp(program);

  auto output_op = testIm2SequenceOp.predict_bn(inputx);
  auto *output_op_ptr = output_op->data<float>();

  auto input_dim = inputx.numel() / inputx.dims()[0];
  DLOG << " input : ";
  for (int i = 0; i < inputx.dims()[0]; ++i) {
    for (int j = 0; j < input_dim; ++j) {
      DLOGF("%f ", inputx_ptr[i * input_dim + j]);
    }
    DLOGF("\n");
  }

  auto output_dim = output_op->numel() / output_op->dims()[0];
  DLOG << " output : ";
  for (int i = 0; i < output_op->dims()[0]; ++i) {
    for (int j = 0; j < output_dim; ++j) {
      DLOGF("%f ", output_op_ptr[i * output_dim + j]);
    }
    DLOGF("\n");
  }

  return 0;
}
