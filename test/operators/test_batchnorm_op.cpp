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
#include "operators/batchnorm_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
class TestBatchNormOp {
 public:
  explicit TestBatchNormOp(const Program<Dtype> p) : program_(p) {
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
        if (op->Type() == "batch_norm" &&
            op->Input("X")[0] == "conv2d_5.tmp_0") {
          DLOG << " mul attr size: " << op->GetAttrMap().size();
          DLOG << " inputs size: " << op->GetInputs().size();
          DLOG << " outputs size: " << op->GetOutputs().size();
          DLOG << " Input X is : " << op->Input("X")[0];
          DLOG << " Input Mean is : " << op->Input("Mean")[0];
          DLOG << " Input Variance is : " << op->Input("Variance")[0];
          DLOG << " Input Scale is : " << op->Input("Scale")[0];
          DLOG << " Input Bias is : " << op->Input("Bias")[0];
          DLOG << " Output Y is : " << op->Output("Y")[0];
          DLOG << " epsilon : " << op->GetAttrMap().at("epsilon").Get<float>();
          std::shared_ptr<operators::BatchNormOp<Dtype, float>> lrn =
              std::make_shared<operators::BatchNormOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope);
          ops_of_block_[*block_desc.get()].push_back(lrn);
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict_bn(const Tensor &t1, const Tensor &t2,
                                     const Tensor &t3, const Tensor &t4,
                                     const Tensor &t5) {
    // feed
    auto scope = program_.scope;
    Variable *x1_feed_value = scope->Var("conv2d_5.tmp_0");
    auto tensor_x1 = x1_feed_value->GetMutable<LoDTensor>();
    tensor_x1->ShareDataWith(t1);

    Variable *mean_feed_value = scope->Var("batch_norm_10.w_1");
    auto tensor_mean = mean_feed_value->GetMutable<LoDTensor>();
    tensor_mean->ShareDataWith(t2);

    Variable *scale_feed_value = scope->Var("batch_norm_10.w_0");
    auto tensor_scale = scale_feed_value->GetMutable<LoDTensor>();
    tensor_scale->ShareDataWith(t3);

    Variable *variance_feed_value = scope->Var("batch_norm_10.w_2");
    auto tensor_variance = variance_feed_value->GetMutable<LoDTensor>();
    tensor_variance->ShareDataWith(t4);

    Variable *bias_feed_value = scope->Var("batch_norm_10.b_0");
    auto tensor_bias = bias_feed_value->GetMutable<LoDTensor>();
    tensor_bias->ShareDataWith(t5);

    Variable *output = scope->Var("batch_norm_10.tmp_2");
    auto *output_tensor = output->GetMutable<LoDTensor>();
    output_tensor->mutable_data<float>({1, 256, 38, 38});
    //  DLOG << typeid(output_tensor).name();
    //  DLOG << "output_tensor dims: " << output_tensor->dims();

    std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
    out_tensor.reset(output_tensor);

    predict_bn(t1, t2, t3, t4, t5, 0);
    return out_tensor;
  }

 private:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;

  void predict_bn(const Tensor &t1, const Tensor &t2, const Tensor &t3,
                  const Tensor &t4, const Tensor &t5, int block_id) {
    std::shared_ptr<BlockDesc> to_predict_block =
        to_predict_program_->Block(block_id);
    for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
      auto op = ops_of_block_[*to_predict_block.get()][j];
      DLOG << "op -> run()";
      op->Run();
    }
  }
};

template class TestBatchNormOp<CPU>;
}  // namespace framework
}  // namespace paddle_mobile

int main() {
  DLOG << "----------**********----------";
  DLOG << "begin to run BatchNormOp Test";
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string(g_mobilenet_ssd));

  /// input x (4,10,2,2)
  paddle_mobile::framework::Tensor inputx1;
  SetupTensor<float>(&inputx1, {1, 256, 38, 38}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx1_ptr = inputx1.data<float>();

  paddle_mobile::framework::Tensor mean;
  SetupTensor<float>(&mean, {256}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *mean_ptr = mean.data<float>();

  paddle_mobile::framework::Tensor scale;
  SetupTensor<float>(&scale, {256}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *scale_ptr = scale.data<float>();

  paddle_mobile::framework::Tensor variance;
  SetupTensor<float>(&variance, {256}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *variance_ptr = variance.data<float>();

  paddle_mobile::framework::Tensor bias;
  SetupTensor<float>(&bias, {256}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *bias_ptr = bias.data<float>();

  paddle_mobile::framework::TestBatchNormOp<paddle_mobile::CPU> testBatchNormOp(
      program);

  auto output_bn =
      testBatchNormOp.predict_bn(inputx1, mean, scale, variance, bias);
  auto *output_bn_ptr = output_bn->data<float>();

  DLOG << " (" << inputx1_ptr[0] << " - " << mean_ptr[0] << ")/(("
       << variance_ptr[0] << " + 0.00001"
       << ")^0.5)* " << scale_ptr[0] << " + " << bias_ptr[0] << " = ";
  DLOG << output_bn_ptr[0];

  DLOG << "input_ptr 0 : " << inputx1_ptr[0];
  DLOG << "output_ptr 0 : " << output_bn_ptr[0];

  return 0;
}
