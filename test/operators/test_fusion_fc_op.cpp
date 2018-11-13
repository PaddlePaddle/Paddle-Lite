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

#include <framework/program/program-optimize/program_optimize.h>
#include "../test_include.h"
#include "operators/fusion_fc_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
class TestFcOp {
 public:
  explicit TestFcOp(const Program<Dtype> p) : program_(p) {
    use_optimize_ = true;
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
        if (op->Type() == "fc" && op->Input("X")[0] == "pool2d_13.tmp_0") {
          DLOG << " fc attr size: " << op->GetAttrMap().size();
          DLOG << " inputs size: " << op->GetInputs().size();
          DLOG << " outputs size: " << op->GetOutputs().size();
          DLOG << " Input X is : " << op->Input("X")[0];
          DLOG << " Input Y is : " << op->Input("Y")[0];
          DLOG << " Input Y is : " << op->Input("Z")[0];
          DLOG << " Output Out is : " << op->Output("Out")[0];
          std::shared_ptr<operators::FusionFcOp<Dtype, float>> testOp =
              std::make_shared<operators::FusionFcOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope);
          ops_of_block_[*block_desc.get()].push_back(testOp);
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict(const Tensor &t1, const Tensor &t2,
                                  const Tensor &t3) {
    // feed
    auto scope = program_.scope;
    Variable *x_feed_value = scope->Var("pool2d_13.tmp_0");
    auto tensor_x = x_feed_value->GetMutable<LoDTensor>();
    tensor_x->ShareDataWith(t1);

    Variable *y_feed_value = scope->Var("loss3_classifier-loc_weights");
    auto tensor_y = y_feed_value->GetMutable<LoDTensor>();
    tensor_y->ShareDataWith(t2);

    Variable *z_feed_value = scope->Var("loss3_classifier-loc_biases");
    auto tensor_z = z_feed_value->GetMutable<LoDTensor>();
    tensor_z->ShareDataWith(t3);

    Variable *con_output = scope->Var("loss3_classifier-loc.tmp_1");
    auto *output_tensor = con_output->GetMutable<LoDTensor>();
    output_tensor->mutable_data<float>({3, 10});
    //  DLOG << typeid(output_tensor).name();
    //  DLOG << "output_tensor dims: " << output_tensor->dims();

    std::shared_ptr<LoDTensor> out_tensor = std::make_shared<LoDTensor>();
    out_tensor.reset(output_tensor);

    predict(t1, t2, t3, 0);
    return out_tensor;
  }

 private:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;

  void predict(const Tensor &t1, const Tensor &t2, const Tensor &t3,
               int block_id) {
    std::shared_ptr<BlockDesc> to_predict_block =
        to_predict_program_->Block(block_id);
    for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
      auto op = ops_of_block_[*to_predict_block.get()][j];
      DLOG << "op -> run()";
      op->Run();
    }
  }
};

template class TestFcOp<CPU>;
}  // namespace framework
}  // namespace paddle_mobile
int main() {
  DLOG << "----------**********----------";
  DLOG << "begin to run Fc Test";
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  //    "../../../test/models/googlenet"
  auto program = loader.Load(g_googlenet);
  paddle_mobile::framework::ProgramOptimize optimize;
  //  program.originProgram->Description("origin");
  auto optimize_program = optimize.FusionOptimize(program.originProgram);

  program.optimizeProgram = optimize_program;

  if (optimize_program != nullptr) {
    optimize_program->Description("optimize");
  } else {
    LOG(paddle_mobile::kLOG_ERROR) << "optimize_program is null";
  }

  /// input x (1,3,224,224)
  paddle_mobile::framework::LoDTensor inputx;
  SetupTensor<float>(&inputx, {3, 64, 1, 1}, static_cast<float>(1),
                     static_cast<float>(1));
  auto *inputx_ptr = inputx.data<float>();
  /// input y (224,)
  paddle_mobile::framework::LoDTensor inputy;
  SetupTensor<float>(&inputy, {64, 10}, static_cast<float>(1.5),
                     static_cast<float>(1.5));
  auto *inputy_ptr = inputy.data<float>();

  paddle_mobile::framework::LoDTensor inputz;
  SetupTensor<float>(&inputz, {10}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputz_ptr = inputz.data<float>();

  paddle_mobile::framework::TestFcOp<paddle_mobile::CPU> testFcOp(program);

  auto output = testFcOp.predict(inputx, inputy, inputz);
  auto *output_ptr = output->data<float>();
  for (int j = 0; j < output->numel(); ++j) {
    DLOG << "value of output: " << output_ptr[j];
  }

  DLOG << "1 (3,64) * 2 (64,10) = 96(3,10)";
  DLOG << "output : 96(3,10) + bias(10)";

  return 0;
}
