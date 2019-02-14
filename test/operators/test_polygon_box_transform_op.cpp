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

#include "../test_include.h"
#include "operators/polygon_box_transform_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
class TestPolygonBoxTransformOp {
 public:
  explicit TestPolygonBoxTransformOp(const Program<Dtype> p) : program_(p) {
    if (use_optimize_) {
      to_predict_program_ = program_.optimizeProgram;
    } else {
      to_predict_program_ = program_.originProgram;
    }
    const std::vector<std::shared_ptr<BlockDesc>> blocks =
        to_predict_program_->Blocks();
    for (auto block_desc : blocks) {
      std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
      for (auto op : ops) {
        if (op->Type() == "polygon_box_transform") {
          DLOG << " attr size: " << op->GetAttrMap().size();
          DLOG << " inputs size: " << op->GetInputs().size();
          DLOG << " input is : " << op->Input("Input")[0];
          input_var_name = op->Input("Input")[0];
          DLOG << " outputs size: " << op->GetOutputs().size();
          DLOG << " output is : " << op->Output("Output")[0];
          output_var_name = op->Output("Output")[0];
          std::shared_ptr<operators::PolygonBoxTransformOp<Dtype, float>>
              op_ptr = std::make_shared<
                  operators::PolygonBoxTransformOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope.get());
          ops_of_block_[*block_desc.get()].push_back(op_ptr);
          return;
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict(const Tensor &t) {
    auto scope = program_.scope.get();
    Variable *input_feed_value = scope->Var(input_var_name);
    auto tensor_input = input_feed_value->GetMutable<LoDTensor>();
    tensor_input->ShareDataWith(t);

    Variable *output = scope->Var(output_var_name);
    auto *output_tensor = output->GetMutable<LoDTensor>();

    std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
    out_tensor.reset(output_tensor);

    predict(t, 0);

    return out_tensor;
  }

 private:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;
  string input_var_name;
  string output_var_name;

  void predict(const Tensor &t, int block_id) {
    std::shared_ptr<BlockDesc> to_predict_block =
        to_predict_program_->Block(block_id);
    for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
      auto op = ops_of_block_[*to_predict_block.get()][j];
      op->Run();
    }
  }
};

template class TestPolygonBoxTransformOp<CPU>;
}  // namespace framework
}  // namespace paddle_mobile

int main() {
  DLOG << "----------**********----------";
  DLOG << "begin to run PolygonBoxTransform Test";
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string(g_ocr));

  paddle_mobile::framework::Tensor input;
  SetupTensor<float>(&input, {1, 8, 1, 2}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *input_ptr = input.data<float>();
  for (int i = 0; i < 16; ++i) {
    *(input_ptr + i) = i;
  }
  DLOG << "input : ";
  for (int i = 0; i < input.numel(); ++i) {
    DLOG << " index " << i << " : " << input_ptr[i];
  }

  paddle_mobile::framework::TestPolygonBoxTransformOp<paddle_mobile::CPU>
      testPolygonBoxTransformOp(program);

  auto output = testPolygonBoxTransformOp.predict(input);
  auto *output_ptr = output->data<float>();

  DLOG << "output : ";
  for (int i = 0; i < output->numel(); ++i) {
    DLOG << " index " << i << " : " << output_ptr[i];
  }
  return 0;
}
